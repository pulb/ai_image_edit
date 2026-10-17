# SPDX-License-Identifier: GPL-3.0-or-later
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageFilter

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

    def patterned(self, size):
        img = Image.new("RGB", size)
        for x in range(size[0]):
            for y in range(size[1]):
                img.putpixel((x, y), ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256))
        return img

    def mask(self, size=(64, 64), box=(24, 24, 40, 40)):
        mask = Image.new("L", size, 0)
        mask.paste(255, box)
        return mask


class CompositeTest(CompositeBase):
    """RescaleMode.ORIGINAL: the result has the generated image's size."""

    def composite(self, orig, gen, mask, feather):
        return imaging.composite_with_soft_transition(orig, gen, mask, feather, RescaleMode.ORIGINAL)

    def test_without_a_mask_the_generated_image_is_returned(self):
        orig, gen = self.save("o.png", self.noisy((64, 64), RED)), self.save("g.png", self.noisy((64, 64), GREEN))
        self.assertEqual(self.composite(orig, gen, None, 3), gen)

    def test_an_empty_mask_returns_the_generated_image(self):
        orig, gen = self.save("o.png", self.noisy((64, 64), RED)), self.save("g.png", self.noisy((64, 64), GREEN))
        empty = self.save("m.png", Image.new("L", (64, 64), 0))
        self.assertEqual(self.composite(orig, gen, empty, 3), gen)

    def test_only_the_masked_region_changes(self):
        same = self.noisy((64, 64), RED)
        orig, gen = self.save("o.png", same), self.save("g.png", same.point(lambda v: 255 - v))
        out = Image.open(self.composite(orig, gen, self.save("m.png", self.mask()), 0))
        self.assertEqual(out.size, (64, 64))
        original = Image.open(orig).convert("RGB")
        for point in ((2, 2), (60, 60), (10, 50)):  # well outside the mask
            self.assertEqual(out.getpixel(point), original.getpixel(point))
        self.assertNotEqual(out.getpixel((32, 32)), original.getpixel((32, 32)))  # inside

    @mock.patch.object(imaging, "apply_color_match", side_effect=lambda img, match, box: img)
    def test_feathering_softens_the_edge(self, _):
        base = Image.new("RGB", (64, 64), (100, 100, 100))
        orig, gen = self.save("o.png", base), self.save("g.png", Image.new("RGB", (64, 64), (200, 200, 200)))
        mask = self.save("m.png", self.mask())
        hard = Image.open(self.composite(orig, gen, mask, 0)).convert("L")
        soft = Image.open(self.composite(orig, gen, mask, 4)).convert("L")
        edge = (23, 32)  # just outside the mask box
        self.assertEqual(hard.getpixel(edge), hard.getpixel((2, 2)))
        self.assertGreater(soft.getpixel(edge), soft.getpixel((2, 2)))

    def test_the_original_is_fitted_to_the_generated_size(self):
        orig = self.save("o.png", self.noisy((32, 32), RED))
        gen = self.save("g.png", self.noisy((64, 64), GREEN))
        out = Image.open(self.composite(orig, gen, self.save("m.png", self.mask()), 2))
        self.assertEqual(out.size, (64, 64))

    def test_an_alpha_mask_uses_its_alpha_channel(self):
        orig, gen = self.save("o.png", self.noisy((64, 64), RED)), self.save("g.png", self.noisy((64, 64), GREEN))
        rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        rgba.paste((255, 255, 255, 255), (24, 24, 40, 40))
        out = Image.open(self.composite(orig, gen, self.save("m.png", rgba), 0))
        self.assertEqual(out.getpixel((2, 2)), Image.open(orig).convert("RGB").getpixel((2, 2)))


class CompositeAtOriginalSizeTest(CompositeBase):
    """RescaleMode.EDIT: the original's size and the pixels outside the edited area are kept."""

    def composite(self, orig, gen, mask, feather=0):
        return Image.open(imaging.composite_with_soft_transition(
            self.save("o.png", orig), self.save("g.png", gen), self.save("m.png", mask), feather, RescaleMode.EDIT,
        )).convert("RGB")

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

    def resampling_log(self, orig, gen, mask):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            out = self.composite(orig, gen, mask)
        return out, log.getvalue()

    @mock.patch.object(imaging, "apply_color_match", side_effect=lambda img, match, box: img)
    def test_an_edit_at_the_same_size_is_not_resampled(self, _):
        orig = self.patterned((64, 64))
        gen = Image.new("RGB", (64, 64), (240, 240, 240))
        out, log = self.resampling_log(orig, gen, self.mask((64, 64), (20, 20, 44, 44)))
        self.assertNotIn("Resampling", log)
        self.assertEqual(out.getpixel((32, 32)), (240, 240, 240))

    def test_an_edit_at_another_size_is_resampled_and_logged(self):
        orig = self.patterned((128, 128))
        gen = Image.new("RGB", (64, 64), (240, 240, 240))
        _, log = self.resampling_log(orig, gen, self.mask((128, 128), (40, 40, 88, 88)))
        self.assertIn("Resampling", log)

    def test_a_smaller_original_keeps_its_size_and_outside_pixels(self):
        orig = self.patterned((32, 24))
        gen = Image.new("RGB", (64, 48), (240, 240, 240))
        mask = self.mask((32, 24), (12, 9, 20, 15))
        out = self.composite(orig, gen, mask)
        self.assertEqual(out.size, (32, 24))
        self.check_untouched_outside(out, orig, (12, 9, 20, 15))
        self.assertNotEqual(out.getpixel((16, 12)), orig.getpixel((16, 12)))

    @mock.patch.object(imaging, "apply_color_match", side_effect=lambda img, match, box: img)
    def test_the_edited_area_comes_from_the_generated_image_at_the_right_place(self, _):
        orig = Image.new("RGB", (128, 128), (60, 60, 60))
        gen = Image.new("RGB", (64, 64), (60, 60, 60))
        gen.paste((250, 20, 20), (32, 0, 64, 32))  # top right quadrant is red
        out = self.composite(orig, gen, self.mask((128, 128), (0, 0, 128, 128)))
        self.assertGreater(out.getpixel((100, 20))[0], 200)
        self.assertLess(out.getpixel((20, 100))[0], 100)

    @mock.patch.object(imaging, "apply_color_match", side_effect=lambda img, match, box: img)
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
        with mock.patch.object(imaging, "apply_color_match", side_effect=lambda img, match, box: img):
            hard = self.composite(orig, gen, self.mask((128, 128), (48, 48, 80, 80)), 0).convert("L")
            soft = self.composite(orig, gen, self.mask((128, 128), (48, 48, 80, 80)), 4).convert("L")
        self.assertEqual(hard.getpixel((47, 64)), 100)
        self.assertGreater(soft.getpixel((47, 64)), 100)

    def test_the_edit_lands_exactly_where_it_was_taken_from(self):
        # A model that changes nothing must give back the original: blobs at
        # known places in the original are found at the same places in the
        # result, to within 0.1 px, at the edit's corners as well as its centre
        # (so neither the offset nor the scale is off). Covers enlarging,
        # shrinking and an aspect ratio that the generation size crops.
        for orig_size, gen_size in (((1200, 900), (640, 448)), ((900, 1200), (448, 640)), ((500, 375), (704, 512))):
            with self.subTest(orig=orig_size, gen=gen_size):
                w, h = orig_size
                spots = [(round(w * fx), round(h * fy)) for fx, fy in ((0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2))]
                ys, xs = np.mgrid[0:h, 0:w]
                field = sum(np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * 6.0 ** 2)) for x, y in spots)
                orig = Image.fromarray(np.clip(30 + 200 * field, 0, 255).astype(np.uint8)).convert("RGB")
                orig_path = self.save("o.png", orig)
                with mock.patch("builtins.print"):
                    gen_path = imaging.fit_image_to_size(orig_path, *gen_size, Image.Resampling.LANCZOS)
                mask = self.mask(orig_size, (round(w * 0.05), round(h * 0.05), round(w * 0.95), round(h * 0.95)))
                with mock.patch.object(imaging, "DETAIL_ENLARGEMENT_RAMP", (100.0, 101.0)):
                    out = Image.open(imaging.composite_with_soft_transition(
                        orig_path, gen_path, self.save("m.png", mask), 0, RescaleMode.EDIT,
                    )).convert("L")
                self.assertEqual(out.size, orig_size)
                result = np.asarray(out, dtype=np.float64) - 30
                for x, y in spots:
                    window = result[y - 20:y + 21, x - 20:x + 21].clip(0)
                    wy, wx = np.mgrid[y - 20:y + 21, x - 20:x + 21]
                    self.assertAlmostEqual((window * wx).sum() / window.sum(), x, delta=0.1)
                    self.assertAlmostEqual((window * wy).sum() / window.sum(), y, delta=0.1)

    def test_without_a_mask_or_with_an_empty_mask_the_generated_image_is_returned(self):
        orig, gen = self.save("o2.png", self.noisy((64, 64), RED)), self.save("g2.png", self.noisy((64, 64), GREEN))
        self.assertEqual(imaging.composite_with_soft_transition(orig, gen, None, 3, RescaleMode.EDIT), gen)
        empty = self.save("m2.png", Image.new("L", (64, 64), 0))
        self.assertEqual(imaging.composite_with_soft_transition(orig, gen, empty, 3, RescaleMode.EDIT), gen)


def tinted(img, red=0, blue=0):
    """img with its red and blue channels shifted."""
    r, g, b = img.split()
    return Image.merge("RGB", (r.point(lambda v: max(0, min(255, v + red))), g, b.point(lambda v: max(0, min(255, v + blue)))))


class MaskedColorCorrectionTest(CompositeBase):
    """Masked edits are colour-corrected only when the area outside the mask drifted."""

    def setUp(self):
        super().setUp()
        self.orig = self.patterned((64, 64))
        self.box = (24, 24, 40, 40)

    def edited(self, gen):
        """gen with a wildly different colour in the masked area, as an edit would have."""
        gen = gen.copy()
        gen.paste((250, 240, 10), self.box)
        return gen

    def correct(self, gen, mask=None):
        match = imaging.masked_color_match(gen, self.orig, mask or self.mask((64, 64), self.box))
        return match.corrections if match else None

    def test_no_drift_needs_no_correction(self):
        self.assertIsNone(self.correct(self.edited(self.orig)))

    def test_the_edited_area_does_not_pull_the_colours(self):
        # Only the masked area differs, and it differs hugely: nothing to correct.
        gen = self.orig.copy()
        gen.paste((255, 0, 255), (20, 20, 44, 44))
        self.assertIsNone(self.correct(gen, self.mask((64, 64), (20, 20, 44, 44))))

    def test_a_strong_drift_is_corrected_in_full(self):
        gen = self.edited(tinted(self.orig, red=40, blue=-30))
        with mock.patch.object(imaging, "COLOR_DRIFT_RAMP", (0.0, 1e-6)):
            full = self.correct(gen)
        self.assertEqual(self.correct(gen), full)

    def test_a_drift_within_the_ramp_is_corrected_partly(self):
        gen = self.edited(tinted(self.orig, red=40, blue=-30))
        with mock.patch.object(imaging, "COLOR_DRIFT_RAMP", (0.0, 1e-6)):
            full = self.correct(gen)
        with mock.patch.object(imaging, "COLOR_DRIFT_RAMP", (0.0, 1000.0)):
            partial = self.correct(gen)
        for (shift, scale), (full_shift, full_scale) in zip(partial, full):
            self.assertLess(abs(shift), abs(full_shift) + 1e-9)
            self.assertLess(abs(scale - 1), abs(full_scale - 1) + 1e-9)
        self.assertGreater(abs(partial[1][0]), 0)  # the a channel did shift a little

    def test_a_small_drift_is_left_alone(self):
        self.assertIsNone(self.correct(self.edited(tinted(self.orig, red=1))))

    def test_a_mask_covering_almost_everything_is_not_corrected(self):
        gen = tinted(self.orig, red=60)
        self.assertIsNone(self.correct(gen, self.mask((64, 64), (1, 1, 63, 63))))

    def test_both_composites_remove_the_drift(self):
        gen = self.edited(tinted(self.orig, red=50))
        gen.paste(tinted(self.orig, red=50).crop(self.box), self.box)  # the edit keeps the scene, drifted
        paths = self.save("o.png", self.orig), self.save("g.png", gen), self.save("m.png", self.mask((64, 64), self.box))
        for mode in RescaleMode:
            out = Image.open(imaging.composite_with_soft_transition(*paths, 0, mode)).convert("RGB")
            red_out = sum(out.getpixel((x, y))[0] for x in range(26, 38) for y in range(26, 38))
            red_orig = sum(self.orig.getpixel((x, y))[0] for x in range(26, 38) for y in range(26, 38))
            self.assertLess(abs(red_out - red_orig) / 144, 12, mode)


class LocalColorMatchTest(CompositeBase):
    """Where the model tinted part of the edited area, the edit is matched to the original at the mask's border."""

    SIZE = (128, 128)
    BOX = (32, 32, 96, 96)

    def wall(self):
        """A wall with a gentle gradient and a little texture, like a photographed surface."""
        img = Image.new("RGB", self.SIZE)
        for x in range(self.SIZE[0]):
            for y in range(self.SIZE[1]):
                v = 150 + x // 8 + (x * 3 + y * 5) % 4
                img.putpixel((x, y), (v, v, v + 4))
        return img

    def tint_inside(self, orig, region=None, red=0, grey=0):
        """orig with `region` (inside the mask) tinted, as the model does."""
        gen = orig.copy()
        region = region or self.BOX
        patch = orig.crop(region).point(lambda v: max(0, v - grey))
        gen.paste(tinted(patch, red=red), region[:2])
        return gen

    def match(self, gen, orig=None):
        orig = orig or self.wall()
        return imaging.masked_color_match(gen, orig, self.mask(self.SIZE, self.BOX))

    def corrected_mean(self, gen, match, region, channel=0):
        crop = gen.crop(region)
        out = imaging.apply_color_match(crop, match, region)
        values = [out.getpixel((x, y))[channel] for x in range(out.width) for y in range(out.height)]
        return sum(values) / len(values)

    def test_a_clean_result_needs_no_correction(self):
        orig = self.wall()
        self.assertIsNone(self.match(orig.copy(), orig))

    def test_a_tint_inside_the_edit_is_corrected_locally_only(self):
        orig = self.wall()
        gen = self.tint_inside(orig, grey=25)
        match = self.match(gen, orig)
        self.assertIsNone(match.corrections)  # nothing drifted outside the mask
        self.assertIsNotNone(match.field)

    def test_the_edit_continues_the_original_at_the_border(self):
        orig = self.wall()
        gen = self.tint_inside(orig, grey=25)
        match = self.match(gen, orig)
        region = (34, 34, 94, 40)  # a strip just inside the top border
        tinted_mean = sum(gen.getpixel((x, y))[0] for x in range(34, 94) for y in range(34, 40)) / 360
        original_mean = sum(orig.getpixel((x, y))[0] for x in range(34, 94) for y in range(34, 40)) / 360
        fixed_mean = self.corrected_mean(gen, match, region)
        self.assertLess(abs(fixed_mean - original_mean), abs(tinted_mean - original_mean) / 3)

    def test_the_correction_follows_the_nearest_border(self):
        orig = self.wall()
        gen = self.tint_inside(orig, region=(32, 32, 64, 96), grey=25)  # only the left half is tinted
        match = self.match(gen, orig)
        left = self.corrected_mean(gen, match, (36, 56, 52, 72))
        right = self.corrected_mean(gen, match, (76, 56, 92, 72))
        orig_left = sum(orig.getpixel((x, y))[0] for x in range(36, 52) for y in range(56, 72)) / 256
        orig_right = sum(orig.getpixel((x, y))[0] for x in range(76, 92) for y in range(56, 72)) / 256
        self.assertLess(abs(left - orig_left), 8)  # was 25 off
        self.assertLess(abs(right - orig_right), 5)  # was right all along; only a little of the left tint leaks over

    def test_a_natural_edge_at_the_border_is_left_alone(self):
        # A shadow edge runs exactly along the mask's left border; the model
        # reproduced it, so there is nothing to correct.
        orig = self.wall()
        orig.paste(orig.crop((0, 0, 32, 128)).point(lambda v: v - 20), (0, 0))
        self.assertIsNone(self.match(orig.copy(), orig))

    def test_edited_content_at_the_border_is_not_mistaken_for_a_tint(self):
        orig = self.wall()
        gen = orig.copy()
        gen.paste((20, 20, 20), (32, 32, 96, 96))  # everything inside replaced by something dark
        self.assertIsNone(self.match(gen, orig))

    def test_the_offsets_are_limited(self):
        orig = self.wall()
        gen = self.tint_inside(orig, grey=40)
        with mock.patch.object(imaging, "COLOR_LOCAL_OUTLIER_DE", 1000.0), mock.patch.object(imaging, "COLOR_LOCAL_LIMIT_DE", 2.0):
            match = self.match(gen, orig)
        self.assertLessEqual(float(abs(match.field[:, :, 1:]).max()), 2.0 + 1e-4)
        self.assertLessEqual(float(abs(match.field[:, :, 0]).max()), 2.0 * 2.55 + 1e-4)

    def test_a_mask_without_room_around_it_is_not_corrected_locally(self):
        orig = self.wall()
        gen = orig.copy()
        gen.paste((200, 100, 100), (0, 0, 128, 128))
        full = self.mask(self.SIZE, (0, 0, 128, 128))
        match = imaging.masked_color_match(gen, orig, full)
        self.assertIsNone(match)

    def test_both_composites_use_the_local_match(self):
        orig = self.wall()
        gen = self.tint_inside(orig, grey=25)
        paths = self.save("o.png", orig), self.save("g.png", gen), self.save("m.png", self.mask(self.SIZE, self.BOX))
        for mode in RescaleMode:
            out = Image.open(imaging.composite_with_soft_transition(*paths, 0, mode)).convert("RGB")
            gap = abs(out.getpixel((64, 36))[0] - orig.getpixel((64, 36))[0])
            self.assertLess(gap, 8, mode)


class DetailMatchTest(unittest.TestCase):
    """An enlarged patch is sharpened a little and given the original's grain."""

    BOX = (60, 60, 220, 220)

    def noisy_original(self, sigma=3.0, size=(280, 280)):
        rng = np.random.default_rng(5)
        noise = rng.normal(0, sigma, (size[1], size[0], 1))
        return Image.fromarray(np.clip(128 + noise, 0, 255).astype(np.uint8).repeat(3, axis=2), mode="RGB")

    def smooth_patch(self, size=(160, 160)):
        return Image.new("RGB", size, (128, 128, 128))

    def level(self, img):
        return imaging._noise_level(img)

    def test_a_patch_that_is_not_enlarged_is_left_alone(self):
        patch = self.smooth_patch()
        out = imaging._match_detail(patch, self.noisy_original(), self.BOX, 1.0)
        self.assertEqual(list(out.getdata()), list(patch.getdata()))

    def test_a_smooth_patch_gets_the_originals_grain(self):
        original = self.noisy_original()
        out = imaging._match_detail(self.smooth_patch(), original, self.BOX, 1.7)
        wanted = imaging._noise_level(original.crop(self.BOX))
        self.assertAlmostEqual(self.level(out), wanted, delta=wanted * 0.2)

    def test_the_grain_is_the_same_every_time(self):
        original = self.noisy_original()
        first = imaging._match_detail(self.smooth_patch(), original, self.BOX, 1.7)
        second = imaging._match_detail(self.smooth_patch(), original, self.BOX, 1.7)
        self.assertEqual(list(first.getdata()), list(second.getdata()))

    def test_no_grain_is_added_when_the_patch_is_already_as_noisy(self):
        original = self.noisy_original(sigma=2.0)
        patch = self.noisy_original(sigma=4.0, size=(160, 160))
        out = imaging._match_detail(patch, original, self.BOX, 1.7)
        # only sharpened: still around the patch's own level, nothing added on top
        self.assertLess(self.level(out), self.level(patch) * 1.5)

    def soft_texture(self):
        """Fine detail, softened the way enlarging softens it."""
        patch = Image.new("RGB", (160, 160), (90, 90, 90))
        for x in range(0, 160, 6):
            patch.paste((170, 170, 170), (x, 0, x + 3, 160))
        return patch.filter(ImageFilter.GaussianBlur(1.0))

    def contrast(self, img):
        return float(imaging._luma(img).std())

    def test_sharpening_raises_the_contrast_of_fine_detail(self):
        flat = Image.new("RGB", (280, 280), (128, 128, 128))  # no grain to match, so only sharpening
        out = imaging._match_detail(self.soft_texture(), flat, self.BOX, 1.7)
        self.assertGreater(self.contrast(out), self.contrast(self.soft_texture()) * 1.05)

    def test_the_effect_scales_with_the_enlargement(self):
        flat = Image.new("RGB", (280, 280), (128, 128, 128))
        patch = self.soft_texture()
        partly = imaging._match_detail(patch, flat, self.BOX, 1.3)
        fully = imaging._match_detail(patch, flat, self.BOX, 1.7)
        self.assertGreater(self.contrast(partly), self.contrast(patch))
        self.assertGreater(self.contrast(fully), self.contrast(partly))

    def test_original_size_composites_use_it_for_an_enlarged_patch(self):
        rng = np.random.default_rng(1)
        orig = Image.fromarray(np.clip(128 + rng.normal(0, 3, (400, 400, 1)), 0, 255).astype(np.uint8).repeat(3, axis=2), mode="RGB")
        gen = Image.new("RGB", (200, 200), (128, 128, 128))
        mask = Image.new("L", (400, 400), 0)
        mask.paste(255, (100, 100, 300, 300))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(imaging, "WORK_DIR", Path(tmp)):
            paths = []
            for name, img in (("o", orig), ("g", gen), ("m", mask)):
                paths.append(str(Path(tmp) / f"{name}.png"))
                img.save(paths[-1])
            out = Image.open(imaging.composite_with_soft_transition(*paths, 0, RescaleMode.EDIT)).convert("RGB")
        edited = out.crop((140, 140, 260, 260))
        self.assertGreater(imaging._noise_level(edited), 1.5)  # a flat patch would be 0


class ModelShiftTest(unittest.TestCase):
    """measure_shift() and align_to_input(): textured images, displaced by known amounts."""

    MASK_BOX = (300, 200, 600, 460)

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        fine = Image.fromarray((rng.random((660, 900)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
        coarse = Image.fromarray((rng.random((66, 90)) * 255).astype(np.uint8)).resize((900, 660), Image.Resampling.BICUBIC)
        cls.image = Image.fromarray(((np.asarray(fine, float) + np.asarray(coarse, float)) / 2).astype(np.uint8)).convert("RGB")

    def moved(self, dx=0.0, dy=0.0, zoom=1.0, centre=None):
        cx, cy = centre or (self.image.width / 2, self.image.height / 2)
        matrix = (1 / zoom, 0, cx - (cx + dx) / zoom, 0, 1 / zoom, cy - (cy + dy) / zoom)
        return self.image.transform(self.image.size, Image.Transform.AFFINE, matrix, Image.Resampling.BICUBIC)

    def mask(self):
        mask = Image.new("L", self.image.size, 0)
        mask.paste(255, self.MASK_BOX)
        return mask

    def edited(self, img):
        img = img.copy()
        img.paste((250, 10, 10), self.MASK_BOX)
        return img

    def test_a_shift_is_measured_to_a_tenth_of_a_pixel(self):
        for dx, dy in ((2.5, -1.25), (0.3, 0.0), (-4.0, 3.7), (0.0, 0.0)):
            with self.subTest(dx=dx, dy=dy):
                shift, stretch, _ = imaging._fit_displacement(imaging.measure_shift(self.image, self.moved(dx, dy)))
                self.assertAlmostEqual(shift[0], dx, delta=0.1)
                self.assertAlmostEqual(shift[1], dy, delta=0.1)
                self.assertAlmostEqual(stretch[0], 0.0, delta=0.001)
                self.assertAlmostEqual(stretch[1], 0.0, delta=0.001)

    def test_a_mask_is_measured_around_and_left_out(self):
        result = imaging.measure_shift(self.image, self.edited(self.moved(1.5, -2.0)), self.mask())
        left, top, right, bottom = result["region"]
        self.assertTrue(left < 300 and top < 200 and right > 600 and bottom > 460)
        self.assertTrue(left > 0 or right < 900)  # not the whole image
        centre = [blk for blk in result["blocks"] if blk["coverage"] < imaging.ALIGN_MIN_COVERAGE]
        self.assertEqual(len(centre), 1)  # the block over the mask
        shift, _, _ = imaging._fit_displacement(result)
        self.assertAlmostEqual(shift[0], 1.5, delta=0.1)
        self.assertAlmostEqual(shift[1], -2.0, delta=0.1)

    def test_a_stretch_is_measured(self):
        _, stretch, _ = imaging._fit_displacement(imaging.measure_shift(self.image, self.moved(zoom=1.01)))
        self.assertAlmostEqual(stretch[0], 0.01, delta=0.001)
        self.assertAlmostEqual(stretch[1], 0.01, delta=0.001)

    def test_plain_or_redrawn_blocks_are_skipped(self):
        output = np.asarray(self.moved(-3.0, 0.0)).copy()
        rng = np.random.default_rng(1)
        output[440:, :300] = 30 + rng.normal(0, 2, (220, 300, 1))  # a plain area, redrawn: nothing to match
        result = imaging.measure_shift(self.image, Image.fromarray(output))
        self.assertIsNone(result["blocks"][6]["shift"])  # bottom left
        shift, _, residual = imaging._fit_displacement(result)
        self.assertAlmostEqual(shift[0], -3.0, delta=0.1)
        self.assertLess(residual, 0.2)

    def test_nothing_to_compare_gives_none(self):
        self.assertIsNone(imaging.measure_shift(self.image, self.image, Image.new("L", self.image.size, 255)))

    def align(self, output, mask=None):
        """
        align_to_input() on self.image and `output` around `mask`: (the aligned
        image or None if unchanged, the log). Without a mask an empty one is
        used, which measures blocks over the whole image.
        """
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(imaging, "WORK_DIR", Path(tmp)):
            src, out, mask_path = Path(tmp) / "in.png", Path(tmp) / "out.png", Path(tmp) / "m.png"
            self.image.save(src)
            output.save(out)
            (mask if mask is not None else Image.new("L", self.image.size, 0)).save(mask_path)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                path = imaging.align_to_input(str(src), str(out), str(mask_path))
            aligned = None if path == str(out) else Image.open(path).copy()
        return aligned, log.getvalue()

    def remaining(self, aligned, mask=None):
        result = imaging.measure_shift(self.image, aligned, mask)
        return max(abs(v) for blk in result["blocks"] if blk["shift"] for v in blk["shift"])

    def test_the_log_reports_the_measurement(self):
        _, log = self.align(self.edited(self.moved(2.0, 1.0)), self.mask())
        self.assertIn("around the mask", log)
        self.assertIn("shift (+2.0, +1.0) px", log)

    def test_a_shift_around_the_mask_is_undone(self):
        aligned, log = self.align(self.edited(self.moved(-2.6, 1.4)), self.mask())
        self.assertIn("Corrected", log)
        self.assertLess(self.remaining(aligned, self.mask()), 0.2)

    def test_a_stretch_is_undone(self):
        aligned, log = self.align(self.moved(1.0, 0.0, zoom=1.004))
        self.assertIn("Corrected", log)
        self.assertLess(self.remaining(aligned), 0.2)

    def test_movement_away_from_the_mask_does_not_count(self):
        output = self.moved(-3.0, 0.0)
        output.paste(self.moved(2.0, 0.0).crop((0, 0, 120, 660)), (0, 0))  # the far left drifts the other way
        aligned, log = self.align(self.edited(output), self.mask())
        self.assertIn("Corrected", log)
        self.assertLess(self.remaining(aligned, self.mask()), 0.2)

    def test_slightly_uneven_movement_is_corrected(self):
        # The middle third moves 1 px further than the rest: no fit matches all
        # blocks, but undoing the average brings every block closer.
        output = self.moved(-3.0, 0.0)
        output.paste(self.moved(-4.0, 0.0).crop((0, 220, 900, 440)), (0, 220))
        aligned, log = self.align(output)
        self.assertIn("Corrected", log)
        self.assertLess(self.remaining(aligned), 1.0)

    # Block shifts and correlations measured on a 12 MP selfie (mask over the
    # eyes), as logged, with the region scaled to this test image: blocks that
    # barely moved end up to 0.3 px further off, the rest much closer.
    SELFIE_RUNS = (
        [(-0.1, -0.2, .18), (-1.4, -1.0, .68), (-1.5, -0.8, .57), (-0.5, -0.6, .21), None,
         (-0.8, 0.0, .21), (-0.1, -0.6, .22), (-1.4, 0.0, .67), (-1.9, -0.1, .62)],
        [(-2.0, -0.9, .46), (-3.5, -0.6, .67), (-4.6, -0.1, .47), (-0.7, -1.9, .18), None,
         (-2.5, 0.7, .24), (-1.2, 0.1, .28), (-3.4, 0.7, .69), (-4.4, 0.5, .52)],
    )

    def test_real_uneven_drift_is_corrected(self):
        for blocks in self.SELFIE_RUNS:
            with self.subTest(blocks=blocks[0]):
                xs, ys = np.linspace(100, 800, 4), np.linspace(150, 510, 4)
                result = {"region": (100, 150, 800, 510), "blocks": [{
                    "centre": ((xs[i % 3] + xs[i % 3 + 1]) / 2, (ys[i // 3] + ys[i // 3 + 1]) / 2),
                    "shift": None if b is None else b[:2], "confidence": 0.0 if b is None else b[2],
                    "coverage": 0.0 if b is None else 1.0,
                } for i, b in enumerate(blocks)]}
                with mock.patch.object(imaging, "measure_shift", return_value=result):
                    aligned, log = self.align(self.image, self.mask())
                self.assertIsNotNone(aligned)
                self.assertIn("Corrected", log)

    def test_a_correction_that_moves_a_block_much_further_off_is_refused(self):
        # The top third stays put and the rest moves 3 px: the fit would move the top well off.
        output = self.moved(-3.0, 0.0)
        output.paste(self.image.crop((0, 0, 900, 220)), (0, 0))
        aligned, log = self.align(output)
        self.assertIsNone(aligned)
        self.assertIn("further from where they belong", log)

    def test_a_tiny_shift_is_left_alone(self):
        aligned, log = self.align(self.moved(0.2, -0.1))
        self.assertIsNone(aligned)
        self.assertIn("below", log)

    def test_uneven_movement_is_left_alone(self):
        # The top half moves right and the bottom half left: no shift and stretch explains that.
        output = self.moved(3.0, 0.0)
        output.paste(self.moved(-3.0, 0.0).crop((0, 330, 900, 660)), (0, 330))
        aligned, log = self.align(output)
        self.assertIsNone(aligned)
        self.assertIn("further from where they belong", log)

    def test_an_implausibly_large_shift_is_left_alone(self):
        aligned, log = self.align(self.moved(25.0, 0.0))
        self.assertIsNone(aligned)
        self.assertIn("recomposition", log)

    def test_a_masked_generation_composites_the_aligned_output(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(imaging, "WORK_DIR", Path(tmp)):
            src, out, mask = Path(tmp) / "in.png", Path(tmp) / "out.png", Path(tmp) / "m.png"
            self.image.save(src)
            self.edited(self.moved(2.0, -1.5)).save(out)
            self.mask().save(mask)
            with mock.patch.object(imaging, "composite_with_soft_transition", return_value="/final.png") as composite, \
                    contextlib.redirect_stdout(io.StringIO()):
                imaging.run_generation(str(src), str(mask), *self.image.size, 0, False, lambda path: str(out))
            composited = composite.call_args[0][1]
            self.assertNotEqual(composited, str(out))
            self.assertLess(self.remaining(Image.open(composited), self.mask()), 0.2)


    def test_an_unmasked_generation_is_not_aligned(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(imaging, "WORK_DIR", Path(tmp)):
            src, out = Path(tmp) / "in.png", Path(tmp) / "out.png"
            self.image.save(src)
            self.moved(2.0, -1.5).save(out)
            with mock.patch.object(imaging, "align_to_input") as align:
                result = imaging.run_generation(str(src), None, None, None, 0, False, lambda path: str(out))
            align.assert_not_called()
            self.assertEqual(result, str(out))


class OtherHelpersTest(unittest.TestCase):
    def test_resize_to_cover_and_crop_hits_the_exact_size(self):
        for size in ((100, 50), (50, 100), (33, 77)):
            out = imaging.resize_to_cover_and_crop(Image.new("RGB", size), 64, 48, Image.Resampling.LANCZOS)
            self.assertEqual(out.size, (64, 48))

    def run_generation(self, size, input_size, mask=None):
        """The path infer() receives for a source of `size` that is to be generated at (64, 48)."""
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(imaging, "WORK_DIR", Path(tmp)):
            source = Path(tmp) / "s.png"
            Image.new("RGB", size, RED).save(source)
            received = []
            with mock.patch.object(imaging, "composite_with_soft_transition", return_value="/out.png"):
                imaging.run_generation(
                    str(source), mask, 64, 48, 0, False, lambda path: received.append(path) or str(source), input_size=input_size,
                )
            self.assertEqual(len(received), 1)
            return received[0] == str(source), Image.open(received[0]).size

    def test_the_source_is_fitted_to_the_input_size(self):
        self.assertEqual(self.run_generation((100, 80), (64, 48)), (False, (64, 48)))
        self.assertEqual(self.run_generation((100, 80), (96, 64)), (False, (96, 64)))

    def test_a_masked_source_is_fitted_to_the_generation_size(self):
        self.assertEqual(self.run_generation((100, 80), None, mask="m.png"), (False, (64, 48)))
        self.assertEqual(self.run_generation((100, 80), (64, 48), mask="m.png"), (False, (64, 48)))
        with self.assertRaises(ValueError):
            self.run_generation((100, 80), (96, 64), mask="m.png")

    def test_the_source_is_passed_on_as_it_is_without_an_input_size(self):
        self.assertEqual(self.run_generation((100, 80), None), (True, (100, 80)))

    def test_whole_image_colour_correction_keeps_the_size(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(imaging, "WORK_DIR", Path(tmp)):
            orig, gen = Path(tmp) / "o.png", Path(tmp) / "g.png"
            Image.new("RGB", (16, 16), RED).save(orig)
            Image.new("RGB", (32, 32), GREEN).save(gen)
            out = imaging.apply_whole_image_color_correction(str(orig), str(gen))
            self.assertEqual(Image.open(out).size, (32, 32))

    def test_an_unmasked_generation_is_colour_corrected_only_when_asked(self):
        grey, tinted = (100, 100, 100), (140, 120, 100)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(imaging, "WORK_DIR", Path(tmp)):
            source, output = Path(tmp) / "s.png", Path(tmp) / "o.png"
            Image.new("RGB", (64, 64), grey).save(source)
            Image.new("RGB", (64, 64), tinted).save(output)
            for corrected in (False, True):
                with self.subTest(corrected=corrected):
                    result = imaging.run_generation(str(source), None, None, None, 0, corrected, lambda path: str(output))
                    self.assertEqual(Image.open(result).getpixel((5, 5)), grey if corrected else tinted)


if __name__ == "__main__":
    unittest.main()
