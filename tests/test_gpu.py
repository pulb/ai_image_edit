# SPDX-License-Identifier: GPL-3.0-or-later
import os
import unittest
from unittest import mock

from ai_image_edit.comfyui import gpu

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


class FreeVramTest(unittest.TestCase):
    def run_with(self, env, output="24576\n"):
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"} | env
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(gpu.subprocess, "run", return_value=mock.Mock(stdout=output)) as run:
            return gpu.free_vram_bytes(), run

    def test_first_gpu_by_default(self):
        free, run = self.run_with({})
        self.assertEqual(free, 24 * GIB)
        self.assertNotIn("-i", run.call_args.args[0])

    def test_follows_cuda_visible_devices(self):
        _, run = self.run_with({"CUDA_VISIBLE_DEVICES": "1,2"})
        self.assertEqual(run.call_args.args[0][-2:], ["-i", "1"])

    def test_unknown_when_the_numbering_differs_from_nvidia_smi(self):
        self.assertIsNone(self.run_with({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"})[0])
        self.assertEqual(self.run_with({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})[0], 24 * GIB)

    def test_no_visible_gpu(self):
        for value in ("", "-1"):
            self.assertIsNone(self.run_with({"CUDA_VISIBLE_DEVICES": value})[0])


class EnvironmentTest(unittest.TestCase):
    def test_comfyui_numbers_gpus_like_nvidia_smi(self):
        base = {k: v for k, v in os.environ.items() if k != "CUDA_DEVICE_ORDER"}
        with mock.patch.dict(os.environ, base, clear=True):
            self.assertEqual(gpu.environment()["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        with mock.patch.dict(os.environ, base | {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear=True):
            self.assertEqual(gpu.environment()["CUDA_DEVICE_ORDER"], "FASTEST_FIRST")


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
