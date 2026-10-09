# SPDX-License-Identifier: GPL-3.0-or-later
import os
import unittest
from unittest import mock

from ai_image_edit.models._shared.comfyui import gpu

GIB = gpu.GIB


class DecideTest(unittest.TestCase):
    def test_unknown_free_memory(self):
        args, _ = gpu.decide(10 * GIB, 6, None)
        self.assertEqual(args, [])

    def test_enough_memory(self):
        args, _ = gpu.decide(14 * GIB, 6, 20 * GIB)
        self.assertEqual(args, gpu.FAST_ARGS)

    def test_not_enough_memory(self):
        args, _ = gpu.decide(14 * GIB, 6, 19 * GIB)
        self.assertEqual(args, [])


class ExtraArgsTest(unittest.TestCase):
    def test_env_overrides(self):
        with mock.patch.dict(os.environ, {"COMFY_EXTRA_ARGS": "--lowvram --foo"}):
            args, _ = gpu.comfy_extra_args(1 * GIB)
        self.assertEqual(args, ["--lowvram", "--foo"])

    def test_empty_env_overrides(self):
        with mock.patch.dict(os.environ, {"COMFY_EXTRA_ARGS": ""}), \
                mock.patch.object(gpu, "free_vram_bytes", return_value=80 * GIB):
            args, _ = gpu.comfy_extra_args(1 * GIB)
        self.assertEqual(args, [])

    def test_automatic(self):
        env = {k: v for k, v in os.environ.items() if k != "COMFY_EXTRA_ARGS"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(gpu, "free_vram_bytes", return_value=48 * GIB):
            self.assertEqual(gpu.comfy_extra_args(14 * GIB)[0], gpu.FAST_ARGS)
            self.assertEqual(gpu.comfy_extra_args(14 * GIB, 40)[0], [])


if __name__ == "__main__":
    unittest.main()
