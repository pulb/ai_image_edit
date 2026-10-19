# SPDX-License-Identifier: GPL-3.0-or-later
import os
import unittest
from unittest import mock

from ai_image_edit.core.types import GenerationProgress, ModelCapabilities, ProgressStage, RescaleMode
from ai_image_edit.ui import helpers as common


class StorageSecretTest(unittest.TestCase):
    def secret(self, password, **env):
        base = {k: v for k, v in os.environ.items() if k != "APP_STORAGE_SECRET"}
        with mock.patch.dict(os.environ, base | env, clear=True):
            return common.storage_secret(password)

    def test_stable_for_a_password(self):
        self.assertEqual(self.secret("pw"), self.secret("pw"))

    def test_differs_between_passwords(self):
        self.assertNotEqual(self.secret("pw"), self.secret("pw2"))

    def test_is_salted(self):
        import hashlib
        self.assertNotEqual(self.secret("pw"), hashlib.sha256(b"pw").hexdigest())
        self.assertNotEqual(self.secret("pw"), hashlib.sha256(b"ai-image-edit-session:pw").hexdigest())

    def test_environment_overrides(self):
        self.assertEqual(self.secret("pw", APP_STORAGE_SECRET="abc"), "abc")


class ResolutionTest(unittest.TestCase):
    def test_label_is_the_side_in_k(self):
        self.assertEqual([common.resolution_label(r) for r in (1024, 1536, 2048)], ["1K", "1.5K", "2K"])

    def test_source_gets_the_nearest_resolution(self):
        import tempfile
        from PIL import Image
        from ai_image_edit.sizes import Qwen21Tiers

        cases = {(1024, 768): 1024, (1920, 1080): 1536, (2560, 1440): 2048, (4000, 3000): 2048, (320, 240): 1024}
        with tempfile.TemporaryDirectory() as tmp:
            for size, expected in cases.items():
                path = f"{tmp}/{size[0]}x{size[1]}.png"
                Image.new("RGB", size).save(path)
                self.assertEqual(Qwen21Tiers({}).resolution_for_source(path), expected, size)


class RescaleModeParamTest(unittest.TestCase):
    def params(self, *, inpainting=True, enhancer=False, **overrides):
        caps = ModelCapabilities(
            max_reference_images=1, supports_loras=False, supports_inpainting=inpainting, supports_prompt_enhancer=enhancer,
            supported_aspect_ratios=["Original"], default_aspect_ratio="Original",
            supported_resolutions=[1024], default_resolution=1024,
        )
        kwargs = dict(
            prompt="p", source_image_path="s.png", mask_path="m.png", reference_images=[], seed=1,
            randomize_seed=False, aspect_ratio="Original", resolution=1024, steps=8, lora_files=[],
            apply_color_correction=False,
        )
        return common.params_from_ui(caps, **(kwargs | overrides))

    def test_defaults_to_the_edited_area(self):
        self.assertIs(self.params().rescale_mode, RescaleMode.EDIT)

    def test_takes_an_enum_or_its_value(self):
        self.assertIs(self.params(rescale_mode=RescaleMode.EDIT).rescale_mode, RescaleMode.EDIT)
        self.assertIs(self.params(rescale_mode="edit").rescale_mode, RescaleMode.EDIT)

    def test_unknown_value_is_rejected(self):
        with self.assertRaises(ValueError):
            self.params(rescale_mode="huge")

    def test_detail_matching_is_on_unless_switched_off(self):
        self.assertTrue(self.params().match_detail_enabled)
        self.assertFalse(self.params(match_detail=False).match_detail_enabled)

    def test_the_prompt_is_enhanced_only_if_the_model_can(self):
        self.assertFalse(self.params().enhance_prompt)
        self.assertTrue(self.params(enhancer=True, enhance_prompt=True).enhance_prompt)
        self.assertFalse(self.params(enhancer=False, enhance_prompt=True).enhance_prompt)

    def test_ignored_without_inpainting(self):
        self.assertIs(self.params(inpainting=False, rescale_mode="original").rescale_mode, RescaleMode.EDIT)

    def test_every_mode_has_a_label(self):
        self.assertEqual(set(common.RESCALE_MODE_OPTIONS), {mode.value for mode in RescaleMode})


class AssetsTest(unittest.TestCase):
    def test_placeholders_are_filled_in(self):
        import re
        from ai_image_edit.ui import assets

        for text in (assets.CLIENT_JS, assets.WIDGET_CSS):
            self.assertTrue(text.strip())
            self.assertEqual(re.findall(r"__[A-Z_]+__", text), [])
        self.assertIn(str(assets.DEFAULT_BRUSH_SIZE), assets.CLIENT_JS)


class ProgressTextTest(unittest.TestCase):
    def test_sampling_shows_the_step_and_the_time(self):
        p = GenerationProgress(ProgressStage.SAMPLING, 12, 40)
        self.assertEqual(common.describe_progress(p, 83.9), "Generating: step 12 of 40 · 1:23")
        self.assertAlmostEqual(common.progress_fraction(p), 0.3)

    def test_other_stages_have_an_indeterminate_bar(self):
        p = GenerationProgress(ProgressStage.PREPARING)
        self.assertEqual(common.describe_progress(p, 5), "Preparing… · 0:05")
        self.assertIsNone(common.progress_fraction(p))

    def test_the_prompt_enhancer_shows_its_tokens(self):
        p = GenerationProgress(ProgressStage.ENHANCING_PROMPT, 250)
        self.assertEqual(common.describe_progress(p, 12), "Improving prompt: 250 tokens · 0:12")
        self.assertEqual(common.describe_progress(GenerationProgress(ProgressStage.ENHANCING_PROMPT), 1), "Improving prompt… · 0:01")
        self.assertIsNone(common.progress_fraction(p))

    def test_cancelling_replaces_the_stage(self):
        p = GenerationProgress(ProgressStage.SAMPLING, 3, 40)
        self.assertEqual(common.describe_progress(p, 61, cancelling=True), "Cancelling… · 1:01")


if __name__ == "__main__":
    unittest.main()
