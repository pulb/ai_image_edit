# SPDX-License-Identifier: GPL-3.0-or-later
import dataclasses
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ai_image_edit.core import result_cache
from ai_image_edit.core.types import GenerationParams, RescaleMode


class CacheKeyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.source = self.dir / "source.png"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(self.source)
        self.output = self.dir / "out.png"
        Image.new("RGB", (8, 8)).save(self.output)
        self.calls = 0
        result_cache._cache = result_cache._LastResult()

    def infer(self, path):
        self.calls += 1
        return str(self.output)

    def run_with(self, params):
        wrapped = result_cache.cached_infer(self.infer, "model", params, seed=1)
        return wrapped(str(self.source))

    def params(self, **overrides):
        return dataclasses.replace(
            GenerationParams(prompt="p", source_image_path=str(self.source), mask_path=None, steps=8), **overrides
        )

    def test_the_same_request_reuses_the_result(self):
        self.run_with(self.params())
        self.run_with(self.params())
        self.assertEqual(self.calls, 1)

    def test_what_happens_after_inference_does_not_change_the_key(self):
        self.run_with(self.params())
        self.run_with(self.params(rescale_mode=RescaleMode.EDIT, feather_amount=9, apply_color_correction_enabled=True))
        self.assertEqual(self.calls, 1)

    def test_an_inference_setting_changes_the_key(self):
        self.run_with(self.params())
        self.run_with(self.params(steps=9))
        self.assertEqual(self.calls, 2)


if __name__ == "__main__":
    unittest.main()
