# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

from ai_image_edit.core.paths import WORK_DIR, from_url, to_url


class UrlTest(unittest.TestCase):
    def test_round_trip_with_special_characters(self):
        for name in ("plain.png", "my photo.png", "photo#1.jpg", "100%.png", "a?b&c=d.png", "größe.png"):
            path = str(WORK_DIR / name)
            url = to_url(path)
            self.assertNotIn("#", url)
            self.assertNotIn("?", url)
            self.assertEqual(from_url(url), path)

    def test_absolute_url_from_the_browser(self):
        self.assertEqual(from_url("http://localhost:7860/files/photo%231.jpg"), str(WORK_DIR / "photo#1.jpg"))

    def test_stays_inside_the_work_dir(self):
        self.assertEqual(from_url("/files/..%2F..%2Fetc%2Fpasswd"), str(WORK_DIR / "passwd"))


if __name__ == "__main__":
    unittest.main()
