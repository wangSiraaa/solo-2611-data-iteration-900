"""帧目录服务测试：身份稳定、校验、调色板修订、持久化与原始帧 API。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from animation_pipeline.catalog import CatalogValidationError, FrameCatalog
from animation_pipeline.generator import build_demo_gif


def solid(w: int, h: int, rgb) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, :3] = rgb
    arr[:, :, 3] = 255
    return arr


class CatalogValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cat = FrameCatalog(8, 8, (0, 0, 0))

    def test_rect_out_of_bounds_rejected(self) -> None:
        bad_rects = [(7, 7, 2, 2), (-1, 0, 2, 2), (0, 0, 9, 8)]
        for rect in bad_rects:
            left, top, w, h = rect
            with self.assertRaises(CatalogValidationError):
                self.cat.add_frame(rect, solid(w, h, (1, 2, 3)), 10, 1)
        with self.assertRaises(CatalogValidationError):
            self.cat.add_frame((0, 0, 0, 4), np.zeros((1, 1, 4), np.uint8), 10, 1)

    def test_body_shape_mismatch_rejected(self) -> None:
        with self.assertRaises(CatalogValidationError):
            self.cat.add_frame((0, 0, 3, 3), solid(2, 2, (1, 2, 3)), 10, 1)

    def test_bad_disposal_and_delay_rejected(self) -> None:
        with self.assertRaises(CatalogValidationError):
            self.cat.add_frame((0, 0, 1, 1), solid(1, 1, (0, 0, 0)), 10, 4)
        with self.assertRaises(CatalogValidationError):
            self.cat.add_frame((0, 0, 1, 1), solid(1, 1, (0, 0, 0)), -1, 1)

    def test_palette_index_validation(self) -> None:
        good = tuple((i, i, i) for i in range(4))
        # 索引越界。
        with self.assertRaises(CatalogValidationError):
            self.cat.add_frame((0, 0, 2, 1), solid(2, 1, (0, 0, 0)), 10, 1,
                               palette=good, indices=bytes((0, 5)))
        # 透明索引越界。
        with self.assertRaises(CatalogValidationError):
            self.cat.add_frame((0, 0, 1, 1), solid(1, 1, (0, 0, 0)), 10, 1,
                               palette=good, transparent_index=9, indices=bytes((0,)))


class CatalogIdentityTests(unittest.TestCase):
    def test_ids_stable_and_not_reused_after_delete(self) -> None:
        cat = FrameCatalog(4, 4, (0, 0, 0))
        ids = [cat.add_frame((0, 0, 1, 1), solid(1, 1, (i, i, i)), 10, 1) for i in range(3)]
        self.assertEqual(ids, [0, 1, 2])
        cat.delete_frame(1)
        self.assertEqual(cat.frame_ids, [0, 2])
        new_id = cat.add_frame((0, 0, 1, 1), solid(1, 1, (9, 9, 9)), 10, 1)
        self.assertEqual(new_id, 3)  # 不回收 id=1
        self.assertEqual(cat.frame_ids, [0, 2, 3])
        with self.assertRaises(CatalogValidationError):
            cat.get_frame(1)

    def test_revision_bumps_on_edit(self) -> None:
        cat = FrameCatalog(4, 4, (0, 0, 0))
        fid = cat.add_frame((0, 0, 1, 1), solid(1, 1, (1, 1, 1)), 10, 1)
        r0 = cat.revision
        cat.update_delay(fid, 3)
        self.assertGreater(cat.revision, r0)
        cat.update_disposal(fid, 2)
        cat.replace_body(fid, solid(1, 1, (2, 2, 2)))
        self.assertGreater(cat.revision, r0 + 1)


class CatalogPaletteEditTests(unittest.TestCase):
    def test_update_palette_re_renders_body(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            gif = build_demo_gif(str(Path(d) / "d.gif"))["path"]
            cat = FrameCatalog.from_gif(gif, d)
            # 帧 2 使用局部调色板，索引 3 当前是青。
            fid = cat.frame_ids[2]
            meta = cat.list_metadata()[2]
            self.assertEqual(tuple(meta["palette"][3]), (0, 255, 255))
            new_pal = list(meta["palette"])
            new_pal[3] = (77, 88, 99)
            cat.update_palette(fid, tuple(new_pal), meta["transparent_index"])
            body = cat.get_frame(fid).rgba
            # 索引 3 像素应重渲染为新颜色。
            self.assertEqual(tuple(int(v) for v in body[0, 0]), (77, 88, 99, 255))

    def test_update_palette_rejects_bad_transparent_index(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            gif = build_demo_gif(str(Path(d) / "d.gif"))["path"]
            cat = FrameCatalog.from_gif(gif, d)
            fid = cat.frame_ids[2]
            pal = tuple(cat.list_metadata()[2]["palette"])
            with self.assertRaises(CatalogValidationError):
                cat.update_palette(fid, pal, 999)


class CatalogPersistenceTests(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            gif = build_demo_gif(str(Path(d) / "d.gif"))["path"]
            cat = FrameCatalog.from_gif(gif, Path(d) / "cat")
            before = [f.rgba.copy() for f in cat.iter_raw_frames()]
            ids_before = cat.frame_ids
            reloaded = FrameCatalog.load(Path(d) / "cat")
            self.assertEqual(reloaded.frame_ids, ids_before)
            for i, raw in enumerate(reloaded.iter_raw_frames()):
                self.assertTrue(np.array_equal(raw.rgba, before[i]))
                self.assertEqual(raw.disposal, cat.list_metadata()[i]["disposal"])

    def test_replace_rect_validates_against_canvas(self) -> None:
        cat = FrameCatalog(8, 8, (0, 0, 0))
        fid = cat.add_frame((0, 0, 2, 2), solid(2, 2, (1, 1, 1)), 10, 1)
        cat.replace_rect(fid, (6, 6, 2, 2))  # 同尺寸合法移动
        self.assertEqual(cat.get_frame(fid).rect, (6, 6, 2, 2))
        with self.assertRaises(CatalogValidationError):
            cat.replace_rect(fid, (7, 7, 2, 2))  # 越界
        with self.assertRaises(CatalogValidationError):
            cat.replace_rect(fid, (0, 0, 3, 3))  # 改尺寸但未提供新正文

    def test_publish_to_copies_bodies(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            gif = build_demo_gif(str(Path(d) / "d.gif"))["path"]
            cat = FrameCatalog.from_gif(gif, Path(d) / "v1")
            cat.publish_to(Path(d) / "v2")
            reloaded = FrameCatalog.load(Path(d) / "v2")
            self.assertEqual(reloaded.frame_count(), cat.frame_count())
            self.assertTrue((Path(d) / "v2" / "bodies").exists())


class RawFrameApiTests(unittest.TestCase):
    def test_raw_frame_is_subrect_not_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            gif = build_demo_gif(str(Path(d) / "d.gif"))["path"]
            cat = FrameCatalog.from_gif(gif, None)
            raw = cat.get_frame(cat.frame_ids[1])  # 4x4 子矩形
            self.assertEqual(raw.rgba.shape, (4, 4, 4))
            self.assertEqual((raw.width, raw.height), (4, 4))
            # 原始帧 API 不含画布尺寸的合成结果。
            self.assertNotEqual(raw.rgba.shape[0], cat.height)


if __name__ == "__main__":
    unittest.main()
