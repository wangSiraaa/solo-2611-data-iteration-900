"""替片校核单测试：题面全部验收叙述。

场景：
1. 白逻辑背景透明/画白片：白色饰面允许替片，加红色饰面拒绝并复现红/白差异；
2. 均显示红的处置3/处置1 片：只验显示可换，要求离场同色则不能换；
3. 局部色表一次改稿 + 旧签单复读；
4. 两区 X/Y 柜：批号库存决定能否替片；反例必须真给 X、Y 各一罐红；R 缺货无解；
5. 同 RGB 两批号各自只适配部分区，不能按颜色并库。
"""

from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from animation_pipeline.cabinet import (
    LegacyCabinet,
    PaintCanCabinet,
    PaintLot,
    Region,
)
from animation_pipeline.catalog import FrameCatalog
from animation_pipeline.stripbuilders import (
    R,
    W,
    build_local_palette_strip,
    build_red_pair,
    build_white_pair,
    build_xy_pair,
    revise_local_palette_to_red,
)
from animation_pipeline.filmstrip import Filmstrip
from animation_pipeline.substitution import (
    CheckRequest,
    issue_sheet,
    reopen_sheet,
    verify_substitution,
)

T = Fraction(5, 100)


def white_cabinet(size: int = 1) -> LegacyCabinet:
    return LegacyCabinet(
        size, 1, [Region("P", 0, 0, size, 1)], {"P": [W]}
    )


def white_red_cabinet(size: int = 1) -> LegacyCabinet:
    return LegacyCabinet(
        size, 1, [Region("P", 0, 0, size, 1)], {"P": [W, R]}
    )


class WhiteBackgroundPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strip_a, self.strip_b = build_white_pair(1)
        self.req = [CheckRequest("s0", T)]

    def test_white_finish_allows_substitution(self) -> None:
        report = verify_substitution(
            self.strip_a, self.strip_b, white_cabinet(), self.req
        )
        self.assertTrue(report.success)
        self.assertEqual(report.verdict, "interchangeable")

    def test_adding_red_finish_rejects_and_reproduces_difference(self) -> None:
        report = verify_substitution(
            self.strip_a, self.strip_b, white_red_cabinet(), self.req
        )
        self.assertFalse(report.success)
        self.assertEqual(report.verdict, "rejected")
        mismatch = report.earliest_mismatch
        self.assertEqual(mismatch.coord, (0, 0))
        # 反例是一套真实可装整柜方案：该区入场红。
        self.assertEqual(report.witness_scheme["colors"], {"P": list(R)})
        # 两侧实际颜色即红/白差异。
        self.assertEqual({mismatch.rgb_a, mismatch.rgb_b}, {R, W})
        # 用反例方案真实回放两片，确认差异可复现，不是纸面推断。
        from animation_pipeline.substitution import entry_canvas_from_scheme

        cabinet = white_red_cabinet()
        from animation_pipeline.cabinet import Scheme

        scheme = Scheme(colors={"P": R})
        entry = entry_canvas_from_scheme(cabinet, scheme)
        color_a = tuple(self.strip_a.exit_canvas(entry)[0, 0, :3])
        color_b = tuple(self.strip_b.exit_canvas(entry)[0, 0, :3])
        self.assertEqual((color_a, color_b), (R, W))


class RedPairExitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strip_d3, self.strip_d1 = build_red_pair(1)
        self.req = [CheckRequest("s0", T)]

    def test_display_only_is_interchangeable_on_white(self) -> None:
        report = verify_substitution(
            self.strip_d3, self.strip_d1, white_cabinet(), self.req
        )
        self.assertTrue(report.success)

    def test_exit_equality_requirement_blocks_substitution(self) -> None:
        report = verify_substitution(
            self.strip_d3,
            self.strip_d1,
            white_cabinet(),
            self.req,
            exit_regions=["P"],
        )
        self.assertFalse(report.success)
        mismatch = report.earliest_mismatch
        self.assertEqual(mismatch.stage, "exit")
        # 处置3 离场恢复为入场白；处置1 离场保留红。
        self.assertEqual({mismatch.rgb_a, mismatch.rgb_b}, {W, R})
        self.assertEqual(report.witness_scheme["colors"], {"P": list(W)})

    def test_exit_check_runs_after_all_display_samples(self) -> None:
        # 多个显示采样都一致时，最早不符仍标记为离场而非显示。
        reqs = [CheckRequest(f"s{i}", Fraction(1 + 2 * i, 100)) for i in range(3)]
        report = verify_substitution(
            self.strip_d3, self.strip_d1, white_cabinet(), reqs, exit_regions=["P"]
        )
        self.assertFalse(report.success)
        self.assertEqual(report.earliest_mismatch.stage, "exit")
        self.assertEqual(report.checked_samples, 3)


class LocalPaletteRevisionAndSheetTests(unittest.TestCase):
    def test_revision_changes_body_and_reopen_flags_drift(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cat_dir = Path(d) / "cat"
            catalog, strip = build_local_palette_strip(1, product_dir=cat_dir)
            catalog.save()
            fp_before = strip.fingerprint()
            cabinet = white_cabinet()  # 仅白饰面：改稿前后透明片都可与该片互换

            sheet_path = Path(d) / "sheet.json"
            ref_a, _ = build_white_pair(1)
            report = issue_sheet(
                sheet_path, ref_a, strip, cabinet, [CheckRequest("s0", T)]
            )
            self.assertTrue(report.success)

            # 一次改稿：局部色表索引 1 白->红，按保留索引图重渲染正文并落盘。
            fid = catalog.frame_ids[0]
            revise_local_palette_to_red(catalog, fid)
            catalog.save()
            # 从落盘目录重新载入，确保读到改稿后的真实产品而非内存缓存正文。
            revised = Filmstrip.from_catalog(
                FrameCatalog.load(cat_dir), catalog.frame_ids, "local-palette-revision"
            )
            self.assertNotEqual(revised.fingerprint(), fp_before)
            body = revised.iter_frames()[0].rgba
            self.assertEqual(tuple(int(c) for c in body[0, 0, :3]), R)

            # 重开旧签单：复读旧结论，指纹漂移被标出，并按当前片条重新判定。
            reopened = reopen_sheet(sheet_path, ref_a, revised, cabinet)
            self.assertEqual(reopened["signed_verdict"], "interchangeable")
            self.assertTrue(reopened["drift"]["strip_b"]["changed"])
            self.assertEqual(reopened["drift"]["strip_b"]["signed"], fp_before)
            self.assertEqual(
                reopened["drift"]["strip_b"]["current"], revised.fingerprint()
            )
            self.assertFalse(reopened["drift"]["cabinet"]["changed"])
            self.assertTrue(reopened["anything_changed"])
            # 白柜上：透明片显白、改稿后的红片显红，旧结论被改判，而非盲目沿用。
            self.assertFalse(reopened["verdict_still_holds"])
            self.assertEqual(reopened["current_report"]["verdict"], "rejected")
            mismatch = reopened["current_report"]["earliest_mismatch"]
            self.assertEqual(mismatch["rgb_b"], list(R))

            # 同一份改稿后片条放到白/红柜上同样被拒。
            red_report = verify_substitution(
                ref_a, revised, white_red_cabinet(), [CheckRequest("s0", T)]
            )
            self.assertEqual(red_report.verdict, "rejected")

    def test_revision_to_red_rejected_when_red_finish_present(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            catalog, strip = build_local_palette_strip(1, product_dir=Path(d) / "cat")
            revise_local_palette_to_red(catalog, catalog.frame_ids[0])
            revised = Filmstrip.from_catalog(
                catalog, catalog.frame_ids, "local-palette-revision"
            )
            ref_a, _ = build_white_pair(1)
            report = verify_substitution(
                ref_a, revised, white_red_cabinet(), [CheckRequest("s0", T)]
            )
            self.assertEqual(report.verdict, "rejected")
            self.assertEqual(report.earliest_mismatch.term_b["reason"], "paint")


class TwoRegionCabinetTests(unittest.TestCase):
    def setUp(self) -> None:
        # 2x1 画布：X=(0,0)，Y=(1,0)。
        self.regions = [Region("X", 0, 0, 1, 1), Region("Y", 1, 0, 1, 1)]
        self.strip_a, self.strip_b = build_xy_pair(2, 1)
        self.req = [CheckRequest("s0", T)]

    def _cabinet(self, r_cans: int, w_cans: int = 1) -> PaintCanCabinet:
        lots = [PaintLot("W1", W, w_cans), PaintLot("R1", R, r_cans)]
        return PaintCanCabinet(
            2, 1, self.regions, lots, {"X": ["W1", "R1"], "Y": ["R1"]}
        )

    def test_one_red_can_reserved_for_y_allows_swap(self) -> None:
        cabinet = self._cabinet(r_cans=1, w_cans=1)
        report = verify_substitution(self.strip_a, self.strip_b, cabinet, self.req)
        self.assertTrue(report.success, report.detail)

    def test_two_red_cans_block_swap_with_real_two_region_counterexample(self) -> None:
        cabinet = self._cabinet(r_cans=2, w_cans=1)
        report = verify_substitution(self.strip_a, self.strip_b, cabinet, self.req)
        self.assertEqual(report.verdict, "rejected")
        # 反例必须真的给 X、Y 各一罐红（同一批号 R1 两罐）。
        paints = {p["region"]: p["lot_id"] for p in report.witness_scheme["paints"]}
        self.assertEqual(paints, {"X": "R1", "Y": "R1"})
        self.assertEqual(report.witness_scheme["colors"], {"X": list(R), "Y": list(R)})
        # 用该方案真实回放：X 上还原红/白差异（甲透明留红，乙画白）。
        from animation_pipeline.cabinet import RegionPaint, Scheme

        scheme = Scheme(
            colors={"X": R, "Y": R},
            paints=(
                RegionPaint("X", "R1", R),
                RegionPaint("Y", "R1", R),
            ),
        )
        self.assertTrue(cabinet.supports_scheme(scheme))
        from animation_pipeline.substitution import entry_canvas_from_scheme

        entry = entry_canvas_from_scheme(cabinet, scheme)
        ax = tuple(self.strip_a.exit_canvas(entry)[0, 0, :3])
        bx = tuple(self.strip_b.exit_canvas(entry)[0, 0, :3])
        self.assertEqual((ax, bx), (R, W))
        mismatch = report.earliest_mismatch
        self.assertEqual(mismatch.coord[0], 0)  # X 列
        self.assertEqual(mismatch.region, "X")

    def test_red_stockout_means_cabinet_unsatisfiable_not_success(self) -> None:
        cabinet = self._cabinet(r_cans=0, w_cans=2)
        report = verify_substitution(self.strip_a, self.strip_b, cabinet, self.req)
        self.assertEqual(report.verdict, "cabinet_unsatisfiable")
        self.assertFalse(report.success)
        self.assertIsNone(report.earliest_mismatch)
        self.assertIsNone(report.witness_scheme)

    def test_same_rgb_lots_with_partial_region_fit_not_pooled(self) -> None:
        # RA/RB 同色红：RA 只适配 X，RB 只适配 Y，各一罐。
        lots = [PaintLot("RA", R, 1), PaintLot("RB", R, 1)]
        cabinet = PaintCanCabinet(
            2, 1, self.regions, lots, {"X": ["RA"], "Y": ["RB"]}
        )
        # 两片在该柜可换：X 被 RA 固定为红？不——X 只有红可选，
        # 甲透明显示红、乙画白显示白 -> 必须拒绝，且反例逐区批号正确。
        report = verify_substitution(self.strip_a, self.strip_b, cabinet, self.req)
        self.assertEqual(report.verdict, "rejected")
        paints = {p["region"]: p["lot_id"] for p in report.witness_scheme["paints"]}
        self.assertEqual(paints, {"X": "RA", "Y": "RB"})
        # 若把 RA 砍到缺货，并库思路会拿 RB 顶替 X；正确实现判柜约无解。
        broken_lots = [PaintLot("RA", R, 0), PaintLot("RB", R, 1)]
        broken = PaintCanCabinet(
            2, 1, self.regions, broken_lots, {"X": ["RA"], "Y": ["RB"]}
        )
        report2 = verify_substitution(self.strip_a, self.strip_b, broken, self.req)
        self.assertEqual(report2.verdict, "cabinet_unsatisfiable")


class DurationAndSheetReopenTests(unittest.TestCase):
    def test_unequal_duration_is_input_error(self) -> None:
        strip_a, _ = build_white_pair(1)
        catalog, strip_b = build_local_palette_strip(1)
        catalog.update_delay(strip_b.frame_ids[0], 20)
        strip_b2 = Filmstrip.from_catalog(catalog, catalog.frame_ids, "local-palette-revision")
        report = verify_substitution(
            strip_a, strip_b2, white_cabinet(), [CheckRequest("s0", T)]
        )
        self.assertEqual(report.verdict, "input_error")
        self.assertFalse(report.success)

    def test_sheet_reopen_with_inventory_change(self) -> None:
        regions = [Region("X", 0, 0, 1, 1), Region("Y", 1, 0, 1, 1)]
        strip_a, strip_b = build_xy_pair(2, 1)

        def make_cabinet(r_cans: int) -> PaintCanCabinet:
            return PaintCanCabinet(
                2,
                1,
                regions,
                [PaintLot("W1", W, 1), PaintLot("R1", R, r_cans)],
                {"X": ["W1", "R1"], "Y": ["R1"]},
            )

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sheet.json"
            issue_sheet(path, strip_a, strip_b, make_cabinet(1), [CheckRequest("s0", T)])
            # 只调整一批漆罐供货数（R1: 1 -> 2），重开应识别柜约指纹漂移并改判。
            reopened = reopen_sheet(path, strip_a, strip_b, make_cabinet(2))
            self.assertTrue(reopened["drift"]["cabinet"]["changed"])
            self.assertEqual(reopened["signed_verdict"], "interchangeable")
            self.assertEqual(reopened["current_report"]["verdict"], "rejected")
            self.assertFalse(reopened["verdict_still_holds"])


if __name__ == "__main__":
    unittest.main()
