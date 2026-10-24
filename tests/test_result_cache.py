# SPDX-License-Identifier: GPL-3.0-or-later
import dataclasses
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ai_image_edit.core import result_cache
from ai_image_edit.core.types import Inference


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

    def run_with(self, inference, references=()):
        wrapped = result_cache.cached_infer(self.infer, "model", inference, references, width=64, height=64)
        return wrapped(str(self.source))

    def inference(self, **overrides):
        base = Inference(prompt="p", steps=8, sampler="euler", scheduler="simple", seed=1, options={"cfg": 1.0})
        return dataclasses.replace(base, **overrides)

    def test_the_same_request_reuses_the_result(self):
        self.run_with(self.inference())
        self.run_with(self.inference())
        self.assertEqual(self.calls, 1)

    def test_an_inference_setting_changes_the_key(self):
        self.run_with(self.inference())
        self.run_with(self.inference(steps=9))
        self.assertEqual(self.calls, 2)

    def test_an_option_changes_the_key(self):
        self.run_with(self.inference())
        self.run_with(self.inference(options={"cfg": 2.0}))
        self.assertEqual(self.calls, 2)

    def test_the_reference_pixels_are_part_of_the_key(self):
        reference = self.dir / "ref.png"
        Image.new("RGB", (8, 8), (1, 2, 3)).save(reference)
        self.run_with(self.inference(), [str(reference)])
        self.run_with(self.inference(), [str(reference)])
        Image.new("RGB", (8, 8), (4, 5, 6)).save(reference)
        self.run_with(self.inference(), [str(reference)])
        self.assertEqual(self.calls, 2)


if __name__ == "__main__":
    unittest.main()
