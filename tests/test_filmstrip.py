"""片条回放与像素来源追溯测试。

重点：片条不继承首帧之前的电影内容；处置2 用片条自己的逻辑背景而非入场色；
任意合法入场底画都能回放；来源项可定位到帧与原因。
"""

from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from animation_pipeline.catalog import FrameCatalog
from animation_pipeline.filmstrip import (
    ENTRY_TERM_ID,
    Filmstrip,
    FilmstripValidationError,
    REASON_PAINT,
    REASON_RESTORE_BACKGROUND,
    REASON_RESTORE_BEFORE,
    REASON_TRANSPARENT,
    produce_strip_samples,
)
from animation_pipeline.stripbuilders import R, W, build_white_pair, solid_body


class FilmstripCutTests(unittest.TestCase):
    def test_requires_contiguous_frame_ids_in_order(self) -> None:
        catalog = FrameCatalog(4, 4, W)
        ids = [
            catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, (i, i, i)), 10, 1)
            for i in range(3)
        ]
        # 乱序/跳帧都不允许。
        with self.assertRaises(FilmstripValidationError):
            Filmstrip.from_catalog(catalog, [ids[2], ids[0]], "bad")
        with self.assertRaises(FilmstripValidationError):
            Filmstrip.from_catalog(catalog, [ids[0], ids[2]], "bad")
        strip = Filmstrip.from_catalog(catalog, [ids[1], ids[2]], "ok")
        self.assertEqual(strip.frame_ids, [ids[1], ids[2]])

    def test_carries_logical_canvas_and_background(self) -> None:
        catalog = FrameCatalog(3, 2, (7, 8, 9))
        fid = catalog.add_frame((0, 0, 3, 2), solid_body(3, 2, W), 10, 1)
        strip = Filmstrip.from_catalog(catalog, [fid], "s")
        self.assertEqual((strip.width, strip.height), (3, 2))
        self.assertEqual(strip.background_rgb, (7, 8, 9))


class ArbitraryEntryReplayTests(unittest.TestCase):
    def test_entry_shape_must_match_canvas(self) -> None:
        strip, _ = build_white_pair(1)
        with self.assertRaises(FilmstripValidationError):
            strip.exit_canvas(solid_body(2, 2, W))

    def test_transparent_strip_preserves_arbitrary_entry_color(self) -> None:
        strip, _ = build_white_pair(1)
        # 红底入场：透明一拍不覆盖，显示与离场都仍是红。
        red_entry = solid_body(1, 1, R)
        display = list(strip.replay(red_entry))[0]
        self.assertEqual(tuple(display.canvas[0, 0, :3]), R)
        self.assertEqual(tuple(strip.exit_canvas(red_entry)[0, 0, :3]), R)
        # 白底入场则白——同一条片在两种合法入场底画上结果不同。
        white_entry = solid_body(1, 1, W)
        self.assertEqual(tuple(strip.exit_canvas(white_entry)[0, 0, :3]), W)

    def test_white_paint_strip_is_constant_white_on_any_entry(self) -> None:
        _, strip = build_white_pair(1)
        for rgb in (R, W, (0, 128, 255)):
            out = strip.exit_canvas(solid_body(1, 1, rgb))
            self.assertEqual(tuple(out[0, 0, :3]), W)

    def test_disposal_two_uses_strip_background_not_entry_color(self) -> None:
        # 片条逻辑背景=白；入场底画=红；帧画蓝后处置2，离场必须回到片条白。
        catalog = FrameCatalog(1, 1, W)
        catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, (0, 0, 255)), 10, 2)
        strip = Filmstrip.from_catalog(catalog, catalog.frame_ids, "d2")
        displays = list(strip.replay(solid_body(1, 1, R)))
        self.assertEqual(tuple(displays[0].canvas[0, 0, :3]), (0, 0, 255))
        self.assertEqual(tuple(strip.exit_canvas(solid_body(1, 1, R))[0, 0, :3]), W)

    def test_disposal_three_restores_before_with_arbitrary_entry(self) -> None:
        catalog = FrameCatalog(1, 1, W)
        catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, R), 10, 3)
        strip = Filmstrip.from_catalog(catalog, catalog.frame_ids, "d3")
        # 绿色入场：显示红一拍，处置3 恢复绘制前 -> 离场仍是入场绿。
        entry = solid_body(1, 1, (0, 255, 0))
        self.assertEqual(tuple(list(strip.replay(entry))[0].canvas[0, 0, :3]), R)
        self.assertEqual(tuple(strip.exit_canvas(entry)[0, 0, :3]), (0, 255, 0))

    def test_strip_does_not_inherit_content_before_first_frame(self) -> None:
        # 目录里先有一帧画黑，再切出只含第二帧的片条；片条不得看到那帧黑。
        catalog = FrameCatalog(1, 1, W)
        catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, (0, 0, 0)), 10, 1)
        second = catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, R), 10, 1)
        strip = Filmstrip.from_catalog(catalog, [second], "cut")
        entry = solid_body(1, 1, (9, 9, 9))
        self.assertEqual(tuple(strip.exit_canvas(entry)[0, 0, :3]), R)


class ProvenanceTests(unittest.TestCase):
    def test_transparent_pixel_is_entry_continuation(self) -> None:
        strip, _ = build_white_pair(1)
        _, terms, frame_id = strip.provenance_display(Fraction(5, 100))
        self.assertEqual(frame_id, strip.frame_ids[0])
        term = terms[0][0]
        self.assertTrue(term.is_entry)

    def test_paint_pixel_is_constant_with_located_reason(self) -> None:
        _, strip = build_white_pair(1)
        _, terms, _ = strip.provenance_display(Fraction(5, 100))
        term = terms[0][0]
        self.assertFalse(term.is_entry)
        self.assertEqual(term.rgb, W)
        self.assertEqual(term.reason, REASON_PAINT)
        self.assertEqual(term.frame_id, strip.frame_ids[0])

    def test_explain_pixel_event_chain_covers_all_contracts(self) -> None:
        # 构造 4 帧：透明、画红(d1)、画蓝(d2 背景恢复)、画绿(d3 恢复画前)。
        catalog = FrameCatalog(1, 1, W)
        catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, W, transparent=True), 10, 1)
        catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, R), 10, 1)
        catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, (0, 0, 255)), 10, 2)
        catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, (0, 255, 0)), 10, 3)
        strip = Filmstrip.from_catalog(catalog, catalog.frame_ids, "chain")
        info = strip.explain_pixel((0, 0), at_exit=True)
        events = [e["event"] for e in info["events"]]
        self.assertEqual(
            events,
            [
                REASON_TRANSPARENT,
                REASON_PAINT,
                REASON_PAINT,
                REASON_RESTORE_BACKGROUND,
                REASON_PAINT,
                REASON_RESTORE_BEFORE,
            ],
        )
        # 处置3 恢复到第4帧绘制前（即处置2 留下的片条逻辑背景白）。
        self.assertEqual(info["source"]["rgb"], list(W))
        self.assertEqual(info["source"]["reason"], REASON_RESTORE_BACKGROUND)

    def test_symbolic_trace_independent_of_entry_and_cached(self) -> None:
        strip, _ = build_white_pair(1)
        trace1 = strip.symbolic_trace()
        trace2 = strip.symbolic_trace()
        self.assertIs(trace1, trace2)  # 指纹未变时复用
        self.assertEqual(trace1.display_grids[0].shape, (1, 1))
        self.assertEqual(int(trace1.display_grids[0][0, 0]), ENTRY_TERM_ID)

    def test_duration_matches_effective_delays(self) -> None:
        catalog = FrameCatalog(1, 1, W)
        catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, R), 0, 1)   # 0->1
        catalog.add_frame((0, 0, 1, 1), solid_body(1, 1, R), 10, 1)
        strip = Filmstrip.from_catalog(catalog, catalog.frame_ids, "t")
        self.assertEqual(strip.duration, Fraction(11, 100))

    def test_persistence_roundtrip_keeps_fingerprint(self) -> None:
        strip, _ = build_white_pair(2)
        with tempfile.TemporaryDirectory() as d:
            strip.product_dir = Path(d) / "strip"
            strip.save()
            reloaded = Filmstrip.load(strip.product_dir)
            self.assertEqual(reloaded.fingerprint(), strip.fingerprint())
            entry = solid_body(2, 2, R)
            self.assertTrue(
                np.array_equal(reloaded.exit_canvas(entry), strip.exit_canvas(entry))
            )

    def test_produce_strip_samples_from_arbitrary_entry(self) -> None:
        strip, _ = build_white_pair(2)
        with tempfile.TemporaryDirectory() as d:
            manifest = produce_strip_samples(
                strip, Path(d) / "samples", solid_body(2, 2, R), [Fraction(0), Fraction(1)]
            )
            self.assertEqual(manifest["strip_id"], strip.strip_id)
            import json

            with open(Path(d) / "samples" / "samples.jsonl", encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]
            self.assertFalse(records[0]["ended"])
            # 全透明片：所有像素都是入场延续，常量像素为 0。
            self.assertEqual(records[0]["entry_derived_pixels"], 4)
            self.assertEqual(records[0]["constant_pixels"], 0)
            self.assertTrue(records[1]["ended"])
            self.assertTrue((Path(d) / "samples" / records[0]["png"]).exists())


class SymbolicVsConcreteTests(unittest.TestCase):
    def test_symbolic_evaluation_matches_concrete_replay_pixelwise(self) -> None:
        # 2x2 画布，多帧覆盖四种处置 + 透明 + 子矩形，逐像素交叉验证。
        catalog = FrameCatalog(2, 2, W)
        catalog.add_frame((0, 0, 2, 2), solid_body(2, 2, R), 10, 1)          # 全红 d1
        transparent = solid_body(1, 1, W, transparent=True)
        catalog.add_frame((0, 0, 1, 1), transparent, 10, 1)                  # 透明 d1
        catalog.add_frame((1, 1, 1, 1), solid_body(1, 1, (0, 0, 255)), 10, 2)  # 蓝 d2
        catalog.add_frame((0, 1, 2, 1), solid_body(2, 1, (0, 255, 0)), 10, 3)  # 绿 d3
        strip = Filmstrip.from_catalog(catalog, catalog.frame_ids, "cross")
        trace = strip.symbolic_trace()

        rng_entries = [
            solid_body(2, 2, R),
            solid_body(2, 2, W),
        ]
        # 再构造一块左右异色的入场底画，保证“入场延续”项真的随坐标变化。
        mixed = np.zeros((2, 2, 4), dtype=np.uint8)
        mixed[:, :, :3] = W
        mixed[:, 1, :3] = R
        mixed[:, :, 3] = 255
        rng_entries.append(mixed)

        for entry in rng_entries:
            concrete = list(strip.replay(entry))
            for i, display in enumerate(concrete):
                grid = trace.display_grids[i]
                for y in range(2):
                    for x in range(2):
                        term = trace.term_at(int(grid[y, x]))
                        symbolic_rgb = strip.evaluate_term(term, entry, (x, y))
                        concrete_rgb = tuple(int(c) for c in display.canvas[y, x, :3])
                        self.assertEqual(symbolic_rgb, concrete_rgb)
            # 离场底画同样逐像素一致。
            exit_concrete = strip.exit_canvas(entry)
            for y in range(2):
                for x in range(2):
                    term = trace.term_at(int(trace.exit_grid[y, x]))
                    self.assertEqual(
                        strip.evaluate_term(term, entry, (x, y)),
                        tuple(int(c) for c in exit_concrete[y, x, :3]),
                    )


if __name__ == "__main__":
    unittest.main()
