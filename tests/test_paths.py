# SPDX-License-Identifier: GPL-3.0-or-later
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

from ai_image_edit.core import paths
from ai_image_edit.core.paths import WORK_DIR, new_work_file, to_url, trim_work_dir


class UrlTest(unittest.TestCase):
    def test_special_characters_are_escaped(self):
        for name in ("plain.png", "my photo.png", "photo#1.jpg", "100%.png", "a?b&c=d.png", "größe.png"):
            url = to_url(str(WORK_DIR / name))
            self.assertNotIn("#", url)
            self.assertNotIn("?", url)
            self.assertEqual(unquote(url.removeprefix("/files/")), name)

    def test_new_work_files_are_unique(self):
        a, b = new_work_file("x"), new_work_file("x", ".jpg")
        self.assertNotEqual(a.stem, b.stem)
        self.assertEqual((a.parent, a.suffix, b.suffix), (WORK_DIR, ".png", ".jpg"))
        self.assertTrue(a.name.startswith("x_"))


class WorkMaxSizeTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(paths, "WORK_MAX_BYTES", paths.WORK_MAX_BYTES)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_setting_is_applied(self):
        with mock.patch.dict(os.environ, {"AI_IMAGE_EDIT_WORK_MAX_SIZE": "1k"}):
            paths.read_settings()
        self.assertEqual(paths.WORK_MAX_BYTES, 1024)

    def test_an_empty_setting_is_the_default(self):
        with mock.patch.dict(os.environ, {"AI_IMAGE_EDIT_WORK_MAX_SIZE": " "}), \
                mock.patch.object(paths.shutil, "disk_usage", return_value=mock.Mock(total=2 ** 40)):
            paths.read_settings()
        self.assertEqual(paths.WORK_MAX_BYTES, paths.parse_size(paths.DEFAULT_WORK_MAX_SIZE))

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
