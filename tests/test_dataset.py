"""数据集生产器测试：真实 PNG、JSONL 计数、样本身份、结束态与架构隔离。"""

from __future__ import annotations

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from animation_pipeline.catalog import FrameCatalog
from animation_pipeline.compositor import (
    FramePlayer,
    produce_dataset,
    reference_replay,
    rgba_digest,
)
from animation_pipeline.generator import build_demo_gif


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.gif = build_demo_gif(str(d / "d.gif"))["path"]
        self.catalog = FrameCatalog.from_gif(self.gif, d / "cat")
        self.times = [
            Fraction(0), Fraction(1, 100), Fraction(1, 10), Fraction(3, 20),
            Fraction(11, 50), Fraction(7, 25), Fraction(9, 25), Fraction(2, 5),
            Fraction(1, 2), Fraction(59, 100), Fraction(3, 5), Fraction(100),
        ]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_produces_real_png_and_jsonl(self) -> None:
        out = Path(self.tmp.name) / "ds"
        produce_dataset(self.catalog, out, self.times)
        records = load_jsonl(out / "samples.jsonl")
        self.assertEqual(len(records), len(self.times))
        active = [r for r in records if not r["ended"]]
        for rec in active:
            png_path = out / rec["png"]
            self.assertTrue(png_path.exists())
            with Image.open(png_path) as im:
                arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
            # 回读 PNG 的真实 RGBA 与摘要逐字节一致。
            self.assertEqual(rgba_digest(arr), rec["rgba_digest"])
            self.assertEqual(arr.shape, (8, 8, 4))
        self.assertTrue((out / "manifest.json").exists())

    def test_counts_recomputable_from_raw_frames(self) -> None:
        # 所有覆盖/继承/透明计数必须能仅凭目录原始帧重算，不读数据集产物。
        out = Path(self.tmp.name) / "ds"
        produce_dataset(self.catalog, out, self.times)
        records = load_jsonl(out / "samples.jsonl")
        ref = reference_replay(self.catalog)
        total = self.catalog.width * self.catalog.height
        ref_by_fid = {r["frame_id"]: r for r in ref}
        for rec in records:
            if rec["ended"]:
                continue
            r = ref_by_fid[rec["frame_id"]]
            self.assertEqual(rec["overwritten_pixels"], r["overwritten_pixels"])
            self.assertEqual(rec["transparent_pixels"], r["transparent_pixels"])
            self.assertEqual(
                rec["inherited_pixels"],
                total - r["overwritten_pixels"],
            )
            self.assertEqual(
                rec["overwritten_pixels"] + rec["inherited_pixels"], total
            )

    def test_png_shared_but_sample_identity_distinct(self) -> None:
        # 两个时间命中同一画布（帧0区间内两个点）应共享 PNG，但各有独立 sample_id。
        times = [Fraction(0), Fraction(5, 100), Fraction(1, 10)]
        out = Path(self.tmp.name) / "ds_shared"
        manifest = produce_dataset(self.catalog, out, times)
        records = load_jsonl(out / "samples.jsonl")
        self.assertEqual({r["sample_id"] for r in records},
                         {"s000000", "s000001", "s000002"})
        first_two = records[:2]
        self.assertEqual(first_two[0]["png"], first_two[1]["png"])
        self.assertEqual(first_two[0]["rgba_digest"], first_two[1]["rgba_digest"])
        self.assertNotEqual(first_two[0]["sample_id"], first_two[1]["sample_id"])
        # 第三个点恰在帧0终点，必须切到帧1。
        self.assertEqual(records[2]["frame_id"], self.catalog.frame_ids[1])
        self.assertLess(manifest["unique_png_count"], len(records))

    def test_ended_records_after_end(self) -> None:
        out = Path(self.tmp.name) / "ds_end"
        produce_dataset(self.catalog, out, [Fraction(100), Fraction(1000)])
        records = load_jsonl(out / "samples.jsonl")
        self.assertTrue(all(r["ended"] for r in records))
        self.assertTrue(all(r["png"] is None and r["frame_id"] is None for r in records))

    def test_rational_time_inputs_accepted(self) -> None:
        out = Path(self.tmp.name) / "ds_rat"
        # 接受 Fraction、int、浮点、分数字符串。
        produce_dataset(self.catalog, out, [Fraction(1, 3), 0, "2/25", 0.05])
        records = load_jsonl(out / "samples.jsonl")
        self.assertEqual(len(records), 4)
        self.assertFalse(any(r["ended"] for r in records[:3]))

    def test_editing_early_frame_recomputes_later_frames(self) -> None:
        out1 = Path(self.tmp.name) / "ds_before"
        produce_dataset(self.catalog, out1, [Fraction(2, 10)])
        before = load_jsonl(out1 / "samples.jsonl")[0]

        # 修订帧0背景颜色，后续所有显示画布都应变化。
        first = self.catalog.frame_ids[0]
        new_bg = np.zeros((8, 8, 4), dtype=np.uint8)
        new_bg[:, :, :3] = (10, 20, 30)
        new_bg[:, :, 3] = 255
        self.catalog.replace_body(first, new_bg)

        out2 = Path(self.tmp.name) / "ds_after"
        produce_dataset(self.catalog, out2, [Fraction(2, 10)])
        after = load_jsonl(out2 / "samples.jsonl")[0]
        self.assertNotEqual(before["rgba_digest"], after["rgba_digest"])
        # 帧身份不变，但画布内容已重算。
        self.assertEqual(before["frame_id"], after["frame_id"])

    def test_producer_uses_raw_frame_api_only(self) -> None:
        # 架构约束：生产器模块不得 import GIF 解析器，也不得打开原 GIF。
        import inspect

        import animation_pipeline.compositor as comp

        source = inspect.getsource(comp)
        self.assertNotIn("parse_gif", source)
        self.assertNotIn("import gif_io", source)
        # FramePlayer 只依赖 catalog 的原始帧接口。
        player = FramePlayer(self.catalog)
        self.assertTrue(hasattr(self.catalog, "iter_raw_frames"))
        frames = list(player.replay())
        self.assertEqual(len(frames), self.catalog.frame_count())

    def test_streaming_order_preserves_unsorted_identity(self) -> None:
        # 乱序输入时间点，输出 JSONL 仍按原始请求顺序保留样本身份。
        times = [Fraction(100), Fraction(0), Fraction(1, 10)]
        out = Path(self.tmp.name) / "ds_order"
        produce_dataset(self.catalog, out, times)
        records = load_jsonl(out / "samples.jsonl")
        self.assertEqual([r["sample_id"] for r in records],
                         ["s000000", "s000001", "s000002"])
        self.assertTrue(records[0]["ended"])
        self.assertEqual(records[1]["frame_id"], self.catalog.frame_ids[0])


if __name__ == "__main__":
    unittest.main()
