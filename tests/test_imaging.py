# SPDX-License-Identifier: GPL-3.0-or-later
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from ai_image_edit.core import imaging
from ai_image_edit.core.types import RescaleMode

RED, GREEN = (200, 30, 30), (30, 200, 30)


class CompositeBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        patcher = mock.patch.object(imaging, "WORK_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def save(self, name, image):
        path = self.dir / name
        image.save(path)
        return str(path)

    def noisy(self, size, base):
        """A picture with enough variation for the colour matching to leave it alone."""
        img = Image.new("RGB", size, base)
        for x in range(size[0]):
            for y in range(size[1]):
                if (x + y) % 2:
                    img.putpixel((x, y), tuple(min(255, c + 40) for c in base))
        return img

    def mask(self, size=(64, 64), box=(24, 24, 40, 40)):
        mask = Image.new("L", size, 0)
        mask.paste(255, box)
        return mask


class CompositeTest(CompositeBase):
    def test_without_a_mask_the_generated_image_is_returned(self):
        orig, gen = self.save("o.png", self.noisy((64, 64), RED)), self.save("g.png", self.noisy((64, 64), GREEN))
        self.assertEqual(imaging.composite_with_soft_transition(orig, gen, None, 3), gen)

    def test_an_empty_mask_returns_the_generated_image(self):
        orig, gen = self.save("o.png", self.noisy((64, 64), RED)), self.save("g.png", self.noisy((64, 64), GREEN))
        empty = self.save("m.png", Image.new("L", (64, 64), 0))
        self.assertEqual(imaging.composite_with_soft_transition(orig, gen, empty, 3), gen)

    def test_only_the_masked_region_changes(self):
        same = self.noisy((64, 64), RED)
        orig, gen = self.save("o.png", same), self.save("g.png", same.point(lambda v: 255 - v))
        out = Image.open(imaging.composite_with_soft_transition(orig, gen, self.save("m.png", self.mask()), 0))
        self.assertEqual(out.size, (64, 64))
        original = Image.open(orig).convert("RGB")
        for point in ((2, 2), (60, 60), (10, 50)):  # well outside the mask
            self.assertEqual(out.getpixel(point), original.getpixel(point))
        self.assertNotEqual(out.getpixel((32, 32)), original.getpixel((32, 32)))  # inside

    @mock.patch.object(imaging, "apply_color_correction", side_effect=lambda img, corrections: img)
    def test_feathering_softens_the_edge(self, _):
        base = Image.new("RGB", (64, 64), (100, 100, 100))
        orig, gen = self.save("o.png", base), self.save("g.png", Image.new("RGB", (64, 64), (200, 200, 200)))
        mask = self.save("m.png", self.mask())
        hard = Image.open(imaging.composite_with_soft_transition(orig, gen, mask, 0)).convert("L")
        soft = Image.open(imaging.composite_with_soft_transition(orig, gen, mask, 4)).convert("L")
        edge = (23, 32)  # just outside the mask box
        self.assertEqual(hard.getpixel(edge), hard.getpixel((2, 2)))
        self.assertGreater(soft.getpixel(edge), soft.getpixel((2, 2)))

    def test_the_original_is_fitted_to_the_generated_size(self):
        orig = self.save("o.png", self.noisy((32, 32), RED))
        gen = self.save("g.png", self.noisy((64, 64), GREEN))
        out = Image.open(imaging.composite_with_soft_transition(orig, gen, self.save("m.png", self.mask()), 2))
        self.assertEqual(out.size, (64, 64))

    def test_an_alpha_mask_uses_its_alpha_channel(self):
        orig, gen = self.save("o.png", self.noisy((64, 64), RED)), self.save("g.png", self.noisy((64, 64), GREEN))
        rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        rgba.paste((255, 255, 255, 255), (24, 24, 40, 40))
        out = Image.open(imaging.composite_with_soft_transition(orig, gen, self.save("m.png", rgba), 0))
        self.assertEqual(out.getpixel((2, 2)), Image.open(orig).convert("RGB").getpixel((2, 2)))


class CompositeAtOriginalSizeTest(CompositeBase):
    """RescaleMode.EDIT: the original's size and the pixels outside the edited area are kept."""

    def composite(self, orig, gen, mask, feather=0):
        return Image.open(imaging.composite_with_soft_transition(
            self.save("o.png", orig), self.save("g.png", gen), self.save("m.png", mask), feather, RescaleMode.EDIT,
        )).convert("RGB")

    def patterned(self, size):
        img = Image.new("RGB", size)
        for x in range(size[0]):
            for y in range(size[1]):
                img.putpixel((x, y), ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256))
        return img

    def check_untouched_outside(self, out, orig, box):
        for x in range(0, orig.width, 3):
            for y in range(0, orig.height, 3):
                if not (box[0] <= x < box[2] and box[1] <= y < box[3]):
                    self.assertEqual(out.getpixel((x, y)), orig.getpixel((x, y)), (x, y))

    def test_a_larger_original_keeps_its_size_and_outside_pixels(self):
        orig = self.patterned((128, 96))
        gen = Image.new("RGB", (64, 48), (240, 240, 240))
        mask = self.mask((128, 96), (48, 36, 80, 60))
        out = self.composite(orig, gen, mask)
        self.assertEqual(out.size, (128, 96))
        self.check_untouched_outside(out, orig, (48, 36, 80, 60))
        self.assertNotEqual(out.getpixel((64, 48)), orig.getpixel((64, 48)))

    def test_a_smaller_original_keeps_its_size_and_outside_pixels(self):
        orig = self.patterned((32, 24))
        gen = Image.new("RGB", (64, 48), (240, 240, 240))
        mask = self.mask((32, 24), (12, 9, 20, 15))
        out = self.composite(orig, gen, mask)
        self.assertEqual(out.size, (32, 24))
        self.check_untouched_outside(out, orig, (12, 9, 20, 15))
        self.assertNotEqual(out.getpixel((16, 12)), orig.getpixel((16, 12)))

    @mock.patch.object(imaging, "apply_color_correction", side_effect=lambda img, corrections: img)
    def test_the_edited_area_comes_from_the_generated_image_at_the_right_place(self, _):
        orig = Image.new("RGB", (128, 128), (60, 60, 60))
        gen = Image.new("RGB", (64, 64), (60, 60, 60))
        gen.paste((250, 20, 20), (32, 0, 64, 32))  # top right quadrant is red
        out = self.composite(orig, gen, self.mask((128, 128), (0, 0, 128, 128)))
        self.assertGreater(out.getpixel((100, 20))[0], 200)
        self.assertLess(out.getpixel((20, 100))[0], 100)

    @mock.patch.object(imaging, "apply_color_correction", side_effect=lambda img, corrections: img)
    def test_a_different_aspect_ratio_is_mapped_through_the_cover_crop(self, _):
        # Generated 64x48 covers the original 128x64 scaled by 0.75: the 4:3 frame
        # sees the middle 85 columns of the original, so its edges stay untouched.
        orig = Image.new("RGB", (128, 64), (60, 60, 60))
        gen = Image.new("RGB", (64, 48), (250, 20, 20))
        out = self.composite(orig, gen, self.mask((128, 64), (0, 0, 128, 64)))
        self.assertEqual(out.size, (128, 64))
        self.assertEqual(out.getpixel((0, 32)), (60, 60, 60))
        self.assertEqual(out.getpixel((127, 32)), (60, 60, 60))
        self.assertGreater(out.getpixel((64, 32))[0], 150)

    def test_the_feather_is_scaled_with_the_original(self):
        orig = Image.new("RGB", (128, 128), (100, 100, 100))
        gen = Image.new("RGB", (64, 64), (200, 200, 200))
        with mock.patch.object(imaging, "apply_color_correction", side_effect=lambda img, corrections: img):
            hard = self.composite(orig, gen, self.mask((128, 128), (48, 48, 80, 80)), 0).convert("L")
            soft = self.composite(orig, gen, self.mask((128, 128), (48, 48, 80, 80)), 4).convert("L")
        self.assertEqual(hard.getpixel((47, 64)), 100)
        self.assertGreater(soft.getpixel((47, 64)), 100)

    def test_without_a_mask_or_with_an_empty_mask_the_generated_image_is_returned(self):
        orig, gen = self.save("o2.png", self.noisy((64, 64), RED)), self.save("g2.png", self.noisy((64, 64), GREEN))
        self.assertEqual(imaging.composite_with_soft_transition(orig, gen, None, 3, RescaleMode.EDIT), gen)
        empty = self.save("m2.png", Image.new("L", (64, 64), 0))
        self.assertEqual(imaging.composite_with_soft_transition(orig, gen, empty, 3, RescaleMode.EDIT), gen)


class OtherHelpersTest(unittest.TestCase):
    def test_resize_to_cover_and_crop_hits_the_exact_size(self):
        for size in ((100, 50), (50, 100), (33, 77)):
            out = imaging.resize_to_cover_and_crop(Image.new("RGB", size), 64, 48, Image.Resampling.LANCZOS)
            self.assertEqual(out.size, (64, 48))

    def test_whole_image_colour_correction_keeps_the_size(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(imaging, "WORK_DIR", Path(tmp)):
            orig, gen = Path(tmp) / "o.png", Path(tmp) / "g.png"
            Image.new("RGB", (16, 16), RED).save(orig)
            Image.new("RGB", (32, 32), GREEN).save(gen)
            out = imaging.apply_whole_image_color_correction(str(orig), str(gen))
            self.assertEqual(Image.open(out).size, (32, 32))


if __name__ == "__main__":
    unittest.main()
