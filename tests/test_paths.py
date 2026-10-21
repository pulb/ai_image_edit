# SPDX-License-Identifier: GPL-3.0-or-later
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_image_edit.core import paths
from ai_image_edit.core.paths import WORK_DIR, from_url, to_url, trim_work_dir


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


class WorkMaxSizeTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(paths, "WORK_MAX_BYTES", paths.WORK_MAX_BYTES)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_setting_is_applied(self):
        with mock.patch.dict(os.environ, {"AI_IMAGE_EDIT_WORK_MAX_SIZE": "1k"}):
            paths.read_settings()
        self.assertEqual(paths.WORK_MAX_BYTES, 1024)

    def test_an_invalid_setting_is_an_error(self):
        with mock.patch.dict(os.environ, {"AI_IMAGE_EDIT_WORK_MAX_SIZE": "lots"}):
            with self.assertRaisesRegex(ValueError, "AI_IMAGE_EDIT_WORK_MAX_SIZE='lots' is not a size"):
                paths.read_settings()


class TrimWorkDirTest(unittest.TestCase):
    """Eight 100-byte files, f0 the oldest; a cap of 450 bytes leaves room for four."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for patcher in (mock.patch.object(paths, "WORK_DIR", self.dir), mock.patch.object(paths, "WORK_MAX_BYTES", 450)):
            patcher.start()
            self.addCleanup(patcher.stop)
        for i in range(8):
            path = self.dir / f"f{i}"
            path.write_bytes(b"x" * 100)
            os.utime(path, (1000 + i, 1000 + i))

    def remaining(self):
        return sorted(p.name for p in self.dir.iterdir())

    def test_the_oldest_files_go_first(self):
        trim_work_dir()
        self.assertEqual(self.remaining(), ["f4", "f5", "f6", "f7"])

    def test_the_newest_files_are_kept(self):
        trim_work_dir(keep=6)
        self.assertEqual(self.remaining(), ["f2", "f3", "f4", "f5", "f6", "f7"])

    def test_a_protected_file_survives_however_old_it_is(self):
        trim_work_dir(protect=[str(self.dir / "f0")])
        self.assertEqual(self.remaining(), ["f0", "f5", "f6", "f7"])

    def test_the_folder_stays_above_the_cap_if_only_protected_files_are_left(self):
        trim_work_dir(protect=[str(self.dir / f"f{i}") for i in range(6)])
        self.assertEqual(self.remaining(), ["f0", "f1", "f2", "f3", "f4", "f5"])

    def test_a_protected_path_that_is_not_there_is_ignored(self):
        trim_work_dir(protect=[str(self.dir / "gone.png")])
        self.assertEqual(self.remaining(), ["f4", "f5", "f6", "f7"])


if __name__ == "__main__":
    unittest.main()
