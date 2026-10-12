# SPDX-License-Identifier: GPL-3.0-or-later
import os
import unittest
from unittest import mock

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


class AssetsTest(unittest.TestCase):
    def test_placeholders_are_filled_in(self):
        import re
        from ai_image_edit.ui import assets

        for text in (assets.CLIENT_JS, assets.WIDGET_CSS):
            self.assertTrue(text.strip())
            self.assertEqual(re.findall(r"__[A-Z_]+__", text), [])
        self.assertIn(str(assets.DEFAULT_BRUSH_SIZE), assets.CLIENT_JS)


if __name__ == "__main__":
    unittest.main()
