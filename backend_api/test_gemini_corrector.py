"""Unit tests for GeminiVisionCorrector (no network, no API quota used).

Run from backend_api/:  venv\\Scripts\\python -m unittest test_gemini_corrector -v
"""
import json
import unittest

import numpy as np

from ocr_pipeline import GeminiVisionCorrector


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def answer(entries):
    """A generateContent reply whose text is the JSON list ``entries``."""
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(entries,
                                                                     ensure_ascii=False)}]}}]}


def quota_error(per_day=True):
    quota = "GenerateRequestsPerDayPerProjectPerModel-FreeTier" if per_day else "PerMinute"
    return {"error": {"code": 429, "message": f"Quota exceeded ({quota})",
                      "details": [{"retryDelay": "30000s" if per_day else "1s"}]}}


class FakeHttp:
    """Replies per model from a script: {model: [FakeResponse, ...]}."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def post(self, url, json=None, timeout=None, headers=None):
        model = url.split("/models/")[1].split(":")[0]
        self.calls.append(model)
        return self.script[model].pop(0)


CROP = np.full((40, 200), 255, np.uint8)


def corrector(script, models=("m1", "m2")):
    c = GeminiVisionCorrector("test-key", models=models)
    c._http = FakeHttp(script)
    return c


class AcceptTest(unittest.TestCase):
    def test_spacing_and_dot_fixes_are_accepted(self):
        self.assertEqual(GeminiVisionCorrector._accept("اسنےشکرکیا", "اس نے شکر کیا"),
                         "اس نے شکر کیا")
        self.assertEqual(GeminiVisionCorrector._accept("پاکسنان", " پاکستان "), "پاکستان")

    def test_rewritten_or_invented_text_is_rejected(self):
        self.assertIsNone(GeminiVisionCorrector._accept(
            "پاکسنان ایک خوبصورت ملک ہے", "پاکستان میں انٹرنیٹ بند ہے"))

    def test_empty_and_english_answers_are_rejected(self):
        self.assertIsNone(GeminiVisionCorrector._accept("اردو", ""))
        self.assertIsNone(GeminiVisionCorrector._accept("اردو", "The text says Urdu"))


class RequestTest(unittest.TestCase):
    def test_boxes_are_matched_by_number(self):
        c = corrector({"m1": [FakeResponse(200, answer(
            [{"box": 2, "text": "کتابوں"}, {"box": 1, "text": "پاکستان"}]))]})
        out = c.correct([(CROP, "پاکسنان", "urdu"), (CROP, "کنابوں", "urdu"),
                         (CROP, "اردو", "urdu")])
        self.assertEqual(out, ["پاکستان", "کتابوں", None])   # box 3 missing -> keep OCR
        self.assertEqual(c.last_model, "m1")

    def test_daily_quota_falls_back_to_next_model_and_remembers_it(self):
        c = corrector({"m1": [FakeResponse(429, quota_error())],
                       "m2": [FakeResponse(200, answer([{"box": 1, "text": "پاکستان"}])),
                              FakeResponse(200, answer([{"box": 1, "text": "پاکستان"}]))]})
        self.assertEqual(c.correct([(CROP, "پاکسنان", "urdu")]), ["پاکستان"])
        self.assertEqual(c.last_model, "m2")
        c.correct([(CROP, "پاکسنان", "urdu")])
        self.assertEqual(c._http.calls, ["m1", "m2", "m2"])   # m1 not asked again today

    def test_overloaded_model_is_retried_once_then_skipped(self):
        busy = {"error": {"code": 503, "message": "high demand"}}
        c = corrector({"m1": [FakeResponse(503, busy), FakeResponse(503, busy)],
                       "m2": [FakeResponse(200, answer([{"box": 1, "text": "اردو"}]))]})
        self.assertEqual(c.correct([(CROP, "اردو", "urdu")]), ["اردو"])
        self.assertEqual(c._http.calls, ["m1", "m1", "m2"])

    def test_no_model_left_raises(self):
        c = corrector({"m1": [FakeResponse(429, quota_error())],
                       "m2": [FakeResponse(429, quota_error())]})
        with self.assertRaises(RuntimeError):
            c.correct([(CROP, "اردو", "urdu")])


if __name__ == "__main__":
    unittest.main()
