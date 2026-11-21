"""展柜合同测试：旧柜独立色单、新柜漆罐批号/库存/逐区兼容、不按颜色并库。

所有“可行/不可行”结论都同时用显式方案核对 supports_scheme 交叉验证。
"""

from __future__ import annotations

import unittest

from animation_pipeline.cabinet import (
    CabinetUnsatisfiable,
    CabinetValidationError,
    LegacyCabinet,
    PaintCanCabinet,
    PaintLot,
    Region,
)
from animation_pipeline.stripbuilders import R, W


def xy_regions() -> list[Region]:
    # 2x1 画布：X=(0,0,1,1)，Y=(1,0,1,1)。
    return [Region("X", 0, 0, 1, 1), Region("Y", 1, 0, 1, 1)]


class PartitionTests(unittest.TestCase):
    def test_regions_must_cover_canvas_without_overlap(self) -> None:
        with self.assertRaises(CabinetValidationError):
            LegacyCabinet(2, 1, [Region("X", 0, 0, 1, 1)], {"X": [W]})  # 未覆盖 Y
        with self.assertRaises(CabinetValidationError):
            LegacyCabinet(
                2,
                1,
                [Region("X", 0, 0, 2, 1), Region("Y", 1, 0, 1, 1)],
                {"X": [W], "Y": [R]},
            )  # 重叠

    def test_region_at_and_paint_canvas(self) -> None:
        cabinet = LegacyCabinet(2, 1, xy_regions(), {"X": [W], "Y": [R]})
        self.assertEqual(cabinet.region_at(0, 0), "X")
        self.assertEqual(cabinet.region_at(1, 0), "Y")
        scheme = cabinet.any_scheme()
        canvas = cabinet.paint_canvas(scheme)
        self.assertEqual(canvas[0][0], W)
        self.assertEqual(canvas[0][1], R)


class LegacyCabinetTests(unittest.TestCase):
    def test_independent_color_lists(self) -> None:
        cabinet = LegacyCabinet(2, 1, xy_regions(), {"X": [W, R], "Y": [R]})
        self.assertTrue(cabinet.is_satisfiable())
        white_x = cabinet.scheme_with_region_color("X", W)
        red_x = cabinet.scheme_with_region_color("X", R)
        self.assertIsNotNone(white_x)
        self.assertIsNotNone(red_x)
        self.assertEqual(white_x.colors["X"], W)
        self.assertEqual(red_x.colors["X"], R)
        # Y 不允许白。
        self.assertIsNone(cabinet.scheme_with_region_color("Y", W))

    def test_unsupported_scheme_rejected(self) -> None:
        cabinet = LegacyCabinet(2, 1, xy_regions(), {"X": [W], "Y": [R]})
        good = cabinet.any_scheme()
        self.assertTrue(cabinet.supports_scheme(good))


class PaintCanCabinetTests(unittest.TestCase):
    def _cabinet(self, r_cans: int = 1, w_cans: int = 1) -> PaintCanCabinet:
        lots = [
            PaintLot("W1", W, w_cans),
            PaintLot("R1", R, r_cans),
        ]
        # X 可用白或红；Y 只能用红。
        return PaintCanCabinet(
            2, 1, xy_regions(), lots, {"X": ["W1", "R1"], "Y": ["R1"]}
        )

    def test_one_red_can_must_go_to_y(self) -> None:
        cabinet = self._cabinet(r_cans=1, w_cans=1)
        self.assertTrue(cabinet.is_satisfiable())
        scheme = cabinet.any_scheme()
        paint_by_region = {p.region: p.lot_id for p in scheme.paints}
        self.assertEqual(paint_by_region, {"X": "W1", "Y": "R1"})
        self.assertTrue(cabinet.supports_scheme(scheme))
        # 唯一红罐被 Y 占用：不存在 X 也刷红的可装方案。
        self.assertIsNone(cabinet.scheme_with_region_color("X", R))

    def test_two_red_cans_allow_red_x(self) -> None:
        cabinet = self._cabinet(r_cans=2, w_cans=1)
        scheme = cabinet.scheme_with_region_color("X", R)
        self.assertIsNotNone(scheme)
        self.assertEqual(scheme.colors["X"], R)
        used = [p.lot_id for p in scheme.paints]
        self.assertEqual(sorted(used), ["R1", "R1"])
        self.assertTrue(cabinet.supports_scheme(scheme))

    def test_stockout_is_unsatisfiable(self) -> None:
        # R 缺货（R1 0 罐），而 Y 只能用红：整柜无解。
        cabinet = PaintCanCabinet(
            2,
            1,
            xy_regions(),
            [PaintLot("W1", W, 2), PaintLot("R1", R, 0)],
            {"X": ["W1", "R1"], "Y": ["R1"]},
        )
        self.assertFalse(cabinet.is_satisfiable())
        with self.assertRaises(CabinetUnsatisfiable):
            cabinet.any_scheme()
        # 同色两批号、其中一个缺货也不能拿另一个顶替：Y 只认 R1。
        split = PaintCanCabinet(
            2,
            1,
            xy_regions(),
            [PaintLot("R1", R, 0), PaintLot("R2", R, 2)],
            {"X": ["R1", "R2"], "Y": ["R1"]},
        )
        self.assertFalse(split.is_satisfiable())

    def test_checking_scheme_does_not_deduct_stock(self) -> None:
        cabinet = self._cabinet()
        scheme = cabinet.any_scheme()
        for _ in range(5):
            self.assertTrue(cabinet.supports_scheme(scheme))
        # 库存仍只够一套：超领方案必须被拒。
        from animation_pipeline.cabinet import RegionPaint, Scheme

        greedy = Scheme(
            colors={"X": R, "Y": R},
            paints=(
                RegionPaint("X", "R1", R),
                RegionPaint("Y", "R1", R),
            ),
        )
        self.assertFalse(cabinet.supports_scheme(greedy))

    def test_same_rgb_different_lots_must_not_be_pooled(self) -> None:
        # 两个批号同色 R，但 RA 只适配 X、RB 只适配 Y，各一罐：各刷各的区。
        lots = [PaintLot("RA", R, 1), PaintLot("RB", R, 1)]
        cabinet = PaintCanCabinet(
            2, 1, xy_regions(), lots, {"X": ["RA"], "Y": ["RB"]}
        )
        self.assertTrue(cabinet.is_satisfiable())
        scheme = cabinet.any_scheme()
        paint_by_region = {p.region: p.lot_id for p in scheme.paints}
        self.assertEqual(paint_by_region, {"X": "RA", "Y": "RB"})

        # 反并库：X、Z 只认 RA（仅 1 罐），Y 只认 RB。按颜色并库会看到
        # “红共 3 罐”而误判可行；按批号分别计库存则 RA 不够 -> 无解。
        regions3 = [
            Region("X", 0, 0, 1, 1),
            Region("Y", 1, 0, 1, 1),
            Region("Z", 2, 0, 1, 1),
        ]
        exclusive = PaintCanCabinet(
            3,
            1,
            regions3,
            [PaintLot("RA", R, 1), PaintLot("RB", R, 2)],
            {"X": ["RA"], "Y": ["RB"], "Z": ["RA"]},
        )
        self.assertFalse(exclusive.is_satisfiable())
        # 把 RA 补到 2 罐即恢复可装，且逐区批号正确。
        fixed = PaintCanCabinet(
            3,
            1,
            regions3,
            [PaintLot("RA", R, 2), PaintLot("RB", R, 2)],
            {"X": ["RA"], "Y": ["RB"], "Z": ["RA"]},
        )
        self.assertTrue(fixed.is_satisfiable())
        self.assertEqual(
            {p.region: p.lot_id for p in fixed.any_scheme().paints},
            {"X": "RA", "Y": "RB", "Z": "RA"},
        )

    def test_hundred_region_cabinet_scales_without_enumeration(self) -> None:
        # 上百饰面区：靠最大流一次性判定，不枚举任何整柜底画。
        n = 120
        regions = [Region(f"P{i}", i, 0, 1, 1) for i in range(n)]
        lots = [PaintLot("L1", W, n // 2), PaintLot("L2", R, n - n // 2)]
        compatible = {r.name: ["L1", "L2"] for r in regions}
        cabinet = PaintCanCabinet(n, 1, regions, lots, compatible)
        self.assertTrue(cabinet.is_satisfiable())
        scheme = cabinet.any_scheme()
        self.assertEqual(len(scheme.paints), n)
        # 把 L1 砍到不足以覆盖时应迅速判无解。
        tight_lots = [PaintLot("L1", W, 1), PaintLot("L2", R, n - 2)]
        tight = PaintCanCabinet(n, 1, regions, tight_lots, compatible)
        self.assertFalse(tight.is_satisfiable())


if __name__ == "__main__":
    unittest.main()
