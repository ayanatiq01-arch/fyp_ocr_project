"""Unit tests for Gemini page reading (no network, no API quota used).

Run from backend_api/:  venv\\Scripts\\python -m unittest test_gemini_reader -v
"""
import json
import unittest

import numpy as np

from ocr_pipeline import GeminiPageReader, blocks_from_gemini, script_language

PAGE = {"blocks": [
    {"type": "Title", "rows": [{"cells": [{"box_2d": [10, 300, 50, 700], "text": "اردو  ادب کی"}]}]},
    {"type": "Table", "rows": [
        {"cells": [{"column": 2, "box_2d": [100, 100, 140, 250], "text": "اس نے سنا"},
                   {"column": 0, "box_2d": [100, 800, 140, 950], "text": "سَمِعَ"}]}]},
    {"type": "List", "rows": [
        {"is_bullet": True, "cells": [{"box_2d": [200, 100, 240, 900], "text": "مرزا غالب"}]},
        {"cells": [{"box_2d": [250, 100, 290, 900], "text": ""}]},            # empty: dropped
        {"cells": [{"box_2d": [300, 900, 340, 100], "text": "الحمد لله"}]}]},  # x swapped
    {"type": "Weird", "rows": [{"cells": [{"box_2d": [1, 2], "text": "بلا"}]}]},  # bad box
]}


class BlocksTest(unittest.TestCase):
    def test_structure_boxes_and_columns(self):
        blocks = blocks_from_gemini(PAGE, width=2000, height=1000)
        self.assertEqual([b.type for b in blocks], ["Title", "Table", "List"])
        title, table, lst = blocks
        self.assertEqual(title.rows[0].cells[0].box, (600, 10, 1400, 50))
        self.assertEqual(title.rows[0].cells[0].text, "اردو ادب کی")          # spaces normalised
        self.assertEqual(table.columns, 2)                                  # columns 0,2 -> 0,1
        self.assertEqual([c.text for c in table.rows[0].cells], ["سَمِعَ", "اس نے سنا"])
        self.assertEqual(table.text(), "سَمِعَ\tاس نے سنا")
        self.assertEqual(len(lst.rows), 2)
        self.assertTrue(lst.rows[0].is_bullet)
        self.assertEqual(lst.rows[1].cells[0].box, (200, 300, 1800, 340))  # swapped x fixed

    def test_cells_count_as_read_with_language(self):
        cells = [c for b in blocks_from_gemini(PAGE, 1000, 1000) for r in b.rows for c in r.cells]
        self.assertTrue(all(c.done and c.candidates and c.engine == "Gemini" for c in cells))
        self.assertEqual(cells[0].language, "urdu")       # کی: Urdu-only letters
        self.assertEqual(cells[-1].language, "arabic")    # الحمد لله
        forced = blocks_from_gemini(PAGE, 1000, 1000, language="arabic")
        self.assertTrue(all(c.language == "arabic" for b in forced for r in b.rows for c in r.cells))

    def test_script_language(self):
        self.assertEqual(script_language("پاکستان"), "urdu")
        self.assertEqual(script_language("الحمد لله رب العالمين"), "arabic")
        self.assertEqual(script_language("123"), "unknown")


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def reply(page):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(page, ensure_ascii=False)}]}}]}


def quota_error():
    return {"error": {"code": 429, "message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                      "details": [{"retryDelay": "30000s"}]}}


class FakeHttp:
    def __init__(self, script):
        self.script = script
        self.calls = []

    def post(self, url, json=None, timeout=None, headers=None):
        model = url.split("/models/")[1].split(":")[0]
        self.calls.append(model)
        return self.script[model].pop(0)


IMAGE = np.full((300, 200, 3), 255, np.uint8)


def reader(script):
    r = GeminiPageReader("test-key", models=("m1", "m2"))
    r._http = FakeHttp(script)
    return r


class ReaderTest(unittest.TestCase):
    def test_page_is_parsed(self):
        r = reader({"m1": [FakeResponse(200, reply(PAGE))]})
        self.assertEqual(r.read_page(IMAGE)["blocks"][0]["type"], "Title")
        self.assertEqual(r.last_model, "m1")

    def test_daily_quota_falls_back_and_is_remembered(self):
        r = reader({"m1": [FakeResponse(429, quota_error())],
                    "m2": [FakeResponse(200, reply(PAGE)), FakeResponse(200, reply(PAGE))]})
        r.read_page(IMAGE)
        r.read_page(IMAGE)
        self.assertEqual(r._http.calls, ["m1", "m2", "m2"])
        self.assertEqual(r.last_model, "m2")

    def test_overloaded_model_is_retried_once_then_skipped(self):
        busy = {"error": {"code": 503, "message": "high demand"}}
        r = reader({"m1": [FakeResponse(503, busy), FakeResponse(503, busy)],
                    "m2": [FakeResponse(200, reply(PAGE))]})
        r.read_page(IMAGE)
        self.assertEqual(r._http.calls, ["m1", "m1", "m2"])

    def test_no_model_left_raises(self):
        r = reader({"m1": [FakeResponse(429, quota_error())],
                    "m2": [FakeResponse(429, quota_error())]})
        with self.assertRaises(RuntimeError):
            r.read_page(IMAGE)


if __name__ == "__main__":
    unittest.main()
