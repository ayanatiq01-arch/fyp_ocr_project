"""Unit tests for Gemini page reading (no network, no API quota used).

Run from backend_api/:  venv\\Scripts\\python -m unittest test_gemini_reader -v
"""
import json
import time
import unittest

import numpy as np

import ocr_pipeline
from unittest.mock import patch
from ocr_pipeline import (GeminiPageReader, blocks_from_gemini, blocks_to_markdown,
                          md_escape, script_language)

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


class MarkdownTest(unittest.TestCase):
    def test_page_structure_becomes_markdown(self):
        md = blocks_to_markdown(blocks_from_gemini(PAGE, 1000, 1000))
        self.assertEqual(md, "## اردو ادب کی\n\n"
                             "| سَمِعَ | اس نے سنا |\n| --- | --- |\n\n"
                             "- مرزا غالب\n"
                             "الحمد لله\n")

    def test_lines_keep_hard_breaks(self):
        page = {"blocks": [{"type": "Text", "rows": [
            {"cells": [{"box_2d": [0, 0, 10, 10], "text": "پہلی سطر"}]},
            {"cells": [{"box_2d": [20, 0, 30, 10], "text": "دوسری سطر"}]}]}]}
        self.assertEqual(blocks_to_markdown(blocks_from_gemini(page, 100, 100)),
                         "پہلی سطر  \nدوسری سطر\n")

    def test_markdown_characters_are_escaped(self):
        self.assertEqual(md_escape("# ۱۔ *x* [a]"), r"\# ۱۔ \*x\* \[a\]")
        self.assertEqual(md_escape("1. y"), r"\1. y")
        self.assertEqual(md_escape("a | b", table=True), r"a \| b")


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
    """Replies per model from a script; an entry may be (delay_seconds, response)."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def post(self, url, json=None, timeout=None, headers=None):
        model = url.split("/models/")[1].split(":")[0]
        self.calls.append(model)
        item = self.script[model].pop(0)
        if isinstance(item, tuple):
            time.sleep(item[0])
            item = item[1]
        return item


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

    def test_overloaded_model_hands_over_at_once(self):
        busy = {"error": {"code": 503, "message": "high demand"}}
        r = reader({"m1": [FakeResponse(503, busy)], "m2": [FakeResponse(200, reply(PAGE))]})
        t = time.time()
        r.read_page(IMAGE)
        self.assertEqual(r._http.calls, ["m1", "m2"])        # no retry pause
        self.assertLess(time.time() - t, 1.0)

    def test_slow_model_gets_a_parallel_backup(self):
        r = reader({"m1": [(3.0, FakeResponse(200, reply(PAGE)))],
                    "m2": [FakeResponse(200, reply(PAGE))]})
        with patch.object(ocr_pipeline, "GEMINI_HEDGE_AFTER", 0.3):
            t = time.time()
            r.read_page(IMAGE)
        self.assertEqual(r.last_model, "m2")                  # backup answered first
        self.assertLess(time.time() - t, 2.0)

    def test_cut_off_answer_is_not_used(self):
        cut = reply(PAGE)
        cut["candidates"][0]["finishReason"] = "MAX_TOKENS"
        r = reader({"m1": [FakeResponse(200, cut)], "m2": [FakeResponse(200, reply(PAGE))]})
        r.read_page(IMAGE)
        self.assertEqual(r.last_model, "m2")

    def test_no_model_left_raises(self):
        r = reader({"m1": [FakeResponse(429, quota_error())],
                    "m2": [FakeResponse(429, quota_error())]})
        with self.assertRaises(RuntimeError):
            r.read_page(IMAGE)


if __name__ == "__main__":
    unittest.main()
