"""Unit tests for the rough draft -> Gemini Markdown correction step and the
script routing (no network, no models, no API quota used).

Run from backend_api/:  venv\\Scripts\\python -m unittest test_correction -v
"""
import json
import time
import unittest
from unittest.mock import patch

import numpy as np

import ocr_pipeline
from ocr_pipeline import (Block, Cell, EngineResult, GeminiMarkdownCorrector, Row,
                          blocks_to_markdown, hard_line_breaks, looks_arabic, md_escape)


def cell(text, box=(0, 0, 10, 10), column=0):
    return Cell(box=box, column=column, text=text, language="urdu", engine="UTRNet",
                confidence=0.9, candidates={"urdu": EngineResult("UTRNet", "urdu", text, 0.9)},
                done=True)


def block(kind, rows, columns=1):
    rs = [Row(box=(0, 0, 10, 10), cells=cells, is_bullet=bullet) for cells, bullet in rows]
    return Block(box=(0, 0, 10, 10), type=kind, rows=rs, columns=columns)


class RoutingTest(unittest.TestCase):
    def test_arabic_readings_go_to_easyocr(self):
        self.assertTrue(looks_arabic("لَا رَيْبَ فِيْهِ اِنَّ اللّٰهَ لَا يُخْلِفُ الْمِيْعَادَ"))  # harakat
        self.assertTrue(looks_arabic("الحمد لله رب العالمين في كتابه"))                      # ي ك
        self.assertTrue(looks_arabic("الصلاة والزكاة"))                                       # ة

    def test_urdu_readings_stay_with_utrnet(self):
        self.assertFalse(looks_arabic("اردو زبان برصغیر کی ایک اہم زبان ہے"))
        self.assertFalse(looks_arabic("جن لوگوں نے کفر کا رویہ اختیار کیا ہے"))
        self.assertFalse(looks_arabic("۲۳"))


class MarkdownTest(unittest.TestCase):
    def test_rough_draft_as_markdown(self):
        blocks = [
            block("Title", [([cell("اردو ادب کی")], False)]),
            block("Table", [([cell("سَمِعَ", column=0), cell("اس نے سنا", column=1)], False)], 2),
            block("List", [([cell("مرزا غالب")], True), ([cell("الحمد لله")], False)]),
        ]
        self.assertEqual(blocks_to_markdown(blocks),
                         "## اردو ادب کی\n\n"
                         "| سَمِعَ | اس نے سنا |\n| --- | --- |\n\n"
                         "- مرزا غالب\n"
                         "الحمد لله\n")

    def test_markdown_characters_are_escaped(self):
        self.assertEqual(md_escape("# ۱۔ *x* [a]"), r"\# ۱۔ \*x\* \[a\]")
        self.assertEqual(md_escape("1. y"), r"\1. y")
        self.assertEqual(md_escape("a | b", table=True), r"a \| b")

    def test_book_line_breaks_become_hard_breaks(self):
        md = "## عنوان\n\nپہلی سطر\nدوسری سطر\n\n- ایک\n- دو\n| a | b |"
        self.assertEqual(hard_line_breaks(md),
                         "## عنوان\n\nپہلی سطر  \nدوسری سطر\n\n- ایک\n- دو\n| a | b |")


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def reply(text, reason="STOP"):
    return {"candidates": [{"finishReason": reason, "content": {"parts": [{"text": text}]}}]}


def quota_error():
    return {"error": {"code": 429, "message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                      "details": [{"retryDelay": "30000s"}]}}


class FakeHttp:
    """Replies per model from a script; an entry may be (delay_seconds, response)."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.bodies = []

    def post(self, url, json=None, timeout=None, headers=None):
        model = url.split("/models/")[1].split(":")[0]
        self.calls.append(model)
        self.bodies.append(json)
        item = self.script[model].pop(0)
        if isinstance(item, tuple):
            time.sleep(item[0])
            item = item[1]
        return item


IMAGE = np.full((300, 200, 3), 255, np.uint8)
GOOD = "## اردو ادب\n\nپہلی سطر\nدوسری سطر"


def corrector(script):
    c = GeminiMarkdownCorrector("test-key", models=("m1", "m2"))
    c._http = FakeHttp(script)
    return c


class CorrectorTest(unittest.TestCase):
    def test_request_has_system_prompt_image_and_rough_draft(self):
        c = corrector({"m1": [FakeResponse(200, reply(GOOD))]})
        md = c.correct(IMAGE, "اردو ادپ\nپہلی سطر")
        self.assertEqual(md, "## اردو ادب\n\nپہلی سطر  \nدوسری سطر")
        body = c._http.bodies[0]
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], c.SYSTEM_PROMPT)
        parts = body["contents"][0]["parts"]
        self.assertIn("inline_data", parts[0])                       # the original image
        self.assertIn("اردو ادپ\nپہلی سطر", parts[1]["text"])        # the rough draft

    def test_code_fences_html_tags_and_direction_marks_are_removed(self):
        c = corrector({"m1": [FakeResponse(200, reply(
            "```markdown\n‏## عنوان\n<u>۱۶۱</u> حاشیہ\n```"))]})
        self.assertEqual(c.correct(IMAGE, "x"), "## عنوان\n۱۶۱ حاشیہ")

    def test_bad_answers_hand_over_to_the_next_model(self):
        c = corrector({"m1": [FakeResponse(200, reply("Sorry, I cannot read this."))],
                       "m2": [FakeResponse(200, reply(GOOD))]})
        c.correct(IMAGE, "x")
        self.assertEqual(c.last_model, "m2")
        cut = corrector({"m1": [FakeResponse(200, reply(GOOD, "MAX_TOKENS"))],
                         "m2": [FakeResponse(200, reply(GOOD))]})
        cut.correct(IMAGE, "x")
        self.assertEqual(cut.last_model, "m2")

    def test_daily_quota_falls_back_and_is_remembered(self):
        c = corrector({"m1": [FakeResponse(429, quota_error())],
                       "m2": [FakeResponse(200, reply(GOOD)), FakeResponse(200, reply(GOOD))]})
        c.correct(IMAGE, "x")
        c.correct(IMAGE, "x")
        self.assertEqual(c._http.calls, ["m1", "m2", "m2"])

    def test_overloaded_model_hands_over_at_once(self):
        busy = {"error": {"code": 503, "message": "high demand"}}
        c = corrector({"m1": [FakeResponse(503, busy)], "m2": [FakeResponse(200, reply(GOOD))]})
        t = time.time()
        c.correct(IMAGE, "x")
        self.assertEqual(c._http.calls, ["m1", "m2"])
        self.assertLess(time.time() - t, 1.0)

    def test_slow_model_gets_a_parallel_backup(self):
        c = corrector({"m1": [(3.0, FakeResponse(200, reply(GOOD)))],
                       "m2": [FakeResponse(200, reply(GOOD))]})
        with patch.object(ocr_pipeline, "GEMINI_HEDGE_AFTER", 0.3):
            t = time.time()
            c.correct(IMAGE, "x")
        self.assertEqual(c.last_model, "m2")
        self.assertLess(time.time() - t, 2.0)

    def test_no_model_left_raises(self):
        c = corrector({"m1": [FakeResponse(429, quota_error())],
                       "m2": [FakeResponse(429, quota_error())]})
        with self.assertRaises(RuntimeError):
            c.correct(IMAGE, "x")


if __name__ == "__main__":
    unittest.main()
