"""合成规则测试：处置 0/1/2/3、透明叠层、连续处置3、0延时、时间选择。

所有画布结论都用纯 Python 二维数组参考实现 reference_replay 逐像素核对，
不引用任何图像库的合成结果。
"""

from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np

from animation_pipeline.catalog import FrameCatalog
from animation_pipeline.compositor import (
    CheckpointCache,
    FramePlayer,
    effective_delay_centis,
    reference_replay,
)


def rgba(w: int, h: int, rgb, alpha: int = 255) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, :3] = rgb
    arr[:, :, 3] = alpha
    return arr


class TinyCatalog(FrameCatalog):
    """8x8 内存目录，便于手工构造场景。"""

    def __init__(self, bg=(0, 0, 0)) -> None:
        super().__init__(8, 8, bg)


def px(canvas, x, y):
    return tuple(int(v) for v in canvas[y, x])


class CompositorRulesTests(unittest.TestCase):
    def test_reference_matches_engine_on_demo(self) -> None:
        from animation_pipeline.generator import build_demo_gif
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            gif = build_demo_gif(str(Path(d) / "d.gif"))["path"]
            catalog = FrameCatalog.from_gif(gif, None)
            ref = reference_replay(catalog)
            frames = list(FramePlayer(catalog).replay())
            self.assertEqual(len(ref), len(frames))
            for i, disp in enumerate(frames):
                self.assertEqual(disp.canvas.tolist(), ref[i]["canvas"])
                self.assertEqual(disp.overwritten_pixels, ref[i]["overwritten_pixels"])
                self.assertEqual(disp.transparent_pixels, ref[i]["transparent_pixels"])

    def test_transparent_does_not_overwrite(self) -> None:
        cat = TinyCatalog()
        cat.add_frame((0, 0, 8, 8), rgba(8, 8, (255, 0, 0)), 10, 1)
        body = np.zeros((2, 2, 4), dtype=np.uint8)
        body[:, :, :3] = (0, 255, 0)
        body[:, :, 3] = 255
        body[0, 0, 3] = 0  # 仅左上角透明，其余三个不透明
        fid = cat.add_frame((0, 0, 2, 2), body, 10, 1)
        disp = list(FramePlayer(cat).replay())[1]
        self.assertEqual(disp.frame_id, fid)
        # 透明位置保留上一帧红，三个不透明位置变绿。
        self.assertEqual(px(disp.canvas, 0, 0), (255, 0, 0, 255))
        self.assertEqual(px(disp.canvas, 1, 0), (0, 255, 0, 255))
        self.assertEqual(px(disp.canvas, 0, 1), (0, 255, 0, 255))
        self.assertEqual(px(disp.canvas, 1, 1), (0, 255, 0, 255))
        self.assertEqual(disp.overwritten_pixels, 3)
        self.assertEqual(disp.transparent_pixels, 1)
        self.assertEqual(disp.overwritten_pixels + disp.transparent_pixels, 4)

    def test_disposal_zero_treated_as_one(self) -> None:
        cat = TinyCatalog()
        cat.add_frame((0, 0, 2, 2), rgba(2, 2, (1, 2, 3)), 10, 0)
        cat.add_frame((4, 4, 1, 1), rgba(1, 1, (9, 9, 9)), 10, 1)
        frames = list(FramePlayer(cat).replay())
        # 处置0按1：第二帧画布仍保留第一帧内容。
        self.assertEqual(px(frames[1].canvas, 0, 0), (1, 2, 3, 255))
        self.assertEqual(px(frames[1].canvas, 4, 4), (9, 9, 9, 255))

    def test_disposal_two_restores_subrect_to_background(self) -> None:
        cat = TinyCatalog(bg=(5, 6, 7))
        cat.add_frame((0, 0, 8, 8), rgba(8, 8, (255, 255, 255)), 10, 1)
        cat.add_frame((1, 5, 5, 2), rgba(5, 2, (0, 0, 255)), 10, 2)
        cat.add_frame((0, 0, 1, 1), rgba(1, 1, (255, 0, 255)), 10, 1)
        frames = list(FramePlayer(cat).replay())
        # 第三帧时，第二帧子矩形应被清回背景色，区域外仍是白。
        c2 = frames[2].canvas
        self.assertEqual(px(c2, 1, 5), (5, 6, 7, 255))
        self.assertEqual(px(c2, 5, 6), (5, 6, 7, 255))
        self.assertEqual(px(c2, 0, 0), (255, 0, 255, 255))
        self.assertEqual(px(c2, 7, 7), (255, 255, 255, 255))
        # 但第二帧“显示画布”本身仍是蓝条（先保存再处置）。
        self.assertEqual(px(frames[1].canvas, 1, 5), (0, 0, 255, 255))

    def test_consecutive_disposal_three(self) -> None:
        cat = TinyCatalog(bg=(10, 10, 10))
        cat.add_frame((0, 0, 8, 8), rgba(8, 8, (200, 0, 0)), 10, 1)   # 红底
        cat.add_frame((0, 0, 2, 2), rgba(2, 2, (0, 200, 0)), 10, 3)  # 绿块，恢复
        cat.add_frame((4, 4, 2, 2), rgba(2, 2, (0, 0, 200)), 10, 3)  # 蓝块，恢复
        frames = list(FramePlayer(cat).replay())
        # 两个连续处置3：各自显示后都恢复到绘制前，第三帧后画布回到红底。
        self.assertEqual(px(frames[1].canvas, 0, 0), (0, 200, 0, 255))
        self.assertEqual(px(frames[2].canvas, 4, 4), (0, 0, 200, 255))
        # 蓝帧绘制前应是红底（绿帧已恢复），故蓝帧区域外是红。
        self.assertEqual(px(frames[2].canvas, 0, 0), (200, 0, 0, 255))

    def test_zero_delay_uses_one(self) -> None:
        self.assertEqual(effective_delay_centis(0), 1)
        self.assertEqual(effective_delay_centis(7), 7)
        cat = TinyCatalog()
        cat.add_frame((0, 0, 1, 1), rgba(1, 1, (1, 1, 1)), 0, 1)
        cat.add_frame((1, 1, 1, 1), rgba(1, 1, (2, 2, 2)), 5, 1)
        player = FramePlayer(cat)
        ends = [d.end for d in player.replay()]
        self.assertEqual(ends, [Fraction(1, 100), Fraction(6, 100)])

    def test_half_open_time_selection(self) -> None:
        cat = TinyCatalog()
        ids = [
            cat.add_frame((0, 0, 1, 1), rgba(1, 1, (i, i, i)), 10, 1)
            for i in range(1, 4)
        ]
        player = FramePlayer(cat)
        # 区间 [0,.1) [.1,.2) [.2,.3)，恰在终点选下一帧。
        self.assertEqual(player.select_index(Fraction(0)), 0)
        self.assertEqual(player.select_index(Fraction(99, 1000)), 0)
        self.assertEqual(player.select_index(Fraction(1, 10)), 1)
        self.assertEqual(player.select_index(Fraction(2, 10)), 2)
        # 末尾之后结束，不循环。
        self.assertIsNone(player.select_index(Fraction(3, 10)))
        self.assertIsNone(player.select_index(Fraction(100)))

    def test_no_looping(self) -> None:
        cat = TinyCatalog()
        cat.add_frame((0, 0, 1, 1), rgba(1, 1, (1, 1, 1)), 10, 1)
        player = FramePlayer(cat)
        self.assertEqual(player.end_time, Fraction(1, 10))
        self.assertIsNone(player.select_index(Fraction(1, 10)))

    def test_checkpoint_display_at_matches_replay(self) -> None:
        from animation_pipeline.generator import build_demo_gif
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            gif = build_demo_gif(str(Path(d) / "d.gif"))["path"]
            catalog = FrameCatalog.from_gif(gif, None)
            cache = CheckpointCache(byte_budget=2 * catalog.width * catalog.height * 4)
            player = FramePlayer(catalog, cache)
            replayed = [d.canvas.copy() for d in player.replay()]
            for i in range(catalog.frame_count()):
                got = player.display_at(i).canvas
                self.assertTrue(np.array_equal(got, replayed[i]))

    def test_local_palette_does_not_change_other_regions(self) -> None:
        cat = TinyCatalog()
        cat.add_frame((0, 0, 8, 8), rgba(8, 8, (123, 45, 67)), 10, 1)
        local = tuple((i * 30 % 256, i * 17 % 256, i * 9 % 256) for i in range(8))
        idx = np.zeros((2, 2, 4), dtype=np.uint8)
        idx[:, :, :3] = local[2]
        idx[:, :, 3] = 255
        cat.add_frame((0, 0, 2, 2), idx, 10, 1,
                      palette=local, palette_is_local=True,
                      transparent_index=None, indices=bytes([2] * 4))
        frame1 = list(FramePlayer(cat).replay())[1].canvas
        # 子矩形外颜色完全不变。
        self.assertEqual(px(frame1, 7, 7), (123, 45, 67, 255))
        self.assertEqual(px(frame1, 2, 0), (123, 45, 67, 255))
        self.assertEqual(px(frame1, 0, 0), tuple(local[2]) + (255,))


if __name__ == "__main__":
    unittest.main()
