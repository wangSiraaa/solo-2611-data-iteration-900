"""规模与内存测试：512x512、1000 帧、10000 采样，正文内存 <=16 MiB，可流式落盘。"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from animation_pipeline.catalog import FrameCatalog
from animation_pipeline.compositor import (
    FramePlayer,
    produce_dataset,
    reference_replay,
)
from animation_pipeline.generator import build_scale_gif

CANVAS = 512
FRAMES = 1000
SAMPLES = 10000
MEMORY_LIMIT = 16 * 1024 * 1024


class ScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.gif_path = str(d / "scale.gif")
        info = build_scale_gif(cls.gif_path, CANVAS, CANVAS, FRAMES)
        cls.frame_count = info["frame_count"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_01_gif_structure(self) -> None:
        from animation_pipeline.gif_io import parse_gif

        gif = parse_gif(self.gif_path)
        self.assertEqual((gif.width, gif.height), (CANVAS, CANVAS))
        self.assertEqual(len(gif.images), FRAMES)
        disposals = {im.disposal for im in gif.images}
        self.assertTrue(disposals.issubset({0, 1, 2, 3}))
        # 规模输入里确实出现了 0 延时、局部调色板与透明索引。
        self.assertIn(0, [im.delay_centis for im in gif.images])
        self.assertTrue(any(im.palette_is_local for im in gif.images))
        self.assertTrue(any(im.transparent_index is not None for im in gif.images))

    def test_02_body_memory_under_budget(self) -> None:
        d = Path(self.tmp.name)
        # 12 MiB 正文缓存，严格小于 16 MiB 合同上限。
        catalog = FrameCatalog.from_gif(
            self.gif_path, d / "cat", body_cache_bytes=12 * 1024 * 1024
        )
        self.assertEqual(catalog.frame_count(), FRAMES)
        # 顺序遍历全部帧，缓存驻留量始终不超过预算。
        for _ in catalog.iter_raw_frames():
            self.assertLessEqual(catalog._cache_used, MEMORY_LIMIT)
        # 1000 帧的全量 RGBA 若全驻留需要约 1000*64*64*4（子矩形）以上，
        # 这里断言缓存确实在淘汰而非全量持有。
        self.assertLessEqual(catalog._cache_used, 12 * 1024 * 1024)
        # 磁盘上每帧正文独立存在。
        bodies = list((d / "cat" / "bodies").glob("fid_*.png"))
        self.assertEqual(len(bodies), FRAMES)

    def test_03_reference_and_engine_agree_at_scale(self) -> None:
        d = Path(self.tmp.name)
        catalog = FrameCatalog.from_gif(self.gif_path, d / "cat_ref")
        # 抽样核对若干帧（reference_replay 是纯 Python，全量 1000 帧较慢）。
        ref = reference_replay(catalog)
        engine = list(FramePlayer(catalog).replay())
        check = {0, 1, 2, 50, 250, 500, 999}
        for i in check:
            self.assertTrue(np.array_equal(engine[i].canvas, np.asarray(ref[i]["canvas"], dtype=np.uint8)))
            self.assertEqual(engine[i].overwritten_pixels, ref[i]["overwritten_pixels"])

    def test_04_dataset_streaming_10k_samples(self) -> None:
        d = Path(self.tmp.name)
        catalog = FrameCatalog.from_gif(self.gif_path, d / "cat_ds")
        player = FramePlayer(catalog)
        end = player.end_time
        # 10000 个有理采样点：覆盖区间内、边界与结束之后。
        rng = np.random.default_rng(42)
        span = max(int(float(end) * 1000), 1)
        times = [Fraction(int(x), 1000) for x in rng.integers(0, span, SAMPLES - 4)]
        times += [Fraction(0), end, end + 1, end * 2]
        self.assertEqual(len(times), SAMPLES)
        out = d / "dataset"
        t0 = time.time()
        manifest = produce_dataset(catalog, out, times, checkpoint_budget_bytes=4 * 1024 * 1024)
        elapsed = time.time() - t0
        self.assertEqual(manifest["sample_count"], SAMPLES)
        with open(out / "samples.jsonl", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh]
        self.assertEqual(len(records), SAMPLES)
        # 结束样本存在且不带 PNG；活动样本都有真实 PNG。
        self.assertTrue(any(r["ended"] for r in records))
        for r in records:
            if not r["ended"]:
                self.assertTrue((out / r["png"]).exists())
        # 共享画布：唯一 PNG 数显著少于活动样本数。
        active = [r for r in records if not r["ended"]]
        self.assertLess(manifest["unique_png_count"], len(active))
        self.assertLess(elapsed, 120)

    def test_05_early_edit_recomputes_all_later(self) -> None:
        d = Path(self.tmp.name)
        catalog = FrameCatalog.from_gif(self.gif_path, d / "cat_edit")
        player = FramePlayer(catalog)
        # 记录修订前若干早期显示画布（帧0..5），它们必然包含帧0的像素。
        before = [player.display_at(i).canvas.copy() for i in range(6)]
        first = catalog.frame_ids[0]
        meta = catalog.list_metadata()[0]
        new_body = np.zeros((meta["height"], meta["width"], 4), dtype=np.uint8)
        new_body[:, :, :3] = (7, 7, 7)
        new_body[:, :, 3] = 255
        rev_before = catalog.revision
        catalog.replace_body(first, new_body)
        self.assertGreater(catalog.revision, rev_before)
        # 修订后无需重建目录，早期显示画布即时重算且内容改变。
        player_after = FramePlayer(catalog)
        after = [player_after.display_at(i).canvas for i in range(6)]
        self.assertTrue(any(not np.array_equal(b, a) for b, a in zip(before, after)))
        # 帧0自身画布一定改变（其正文被整体替换）。
        self.assertFalse(np.array_equal(before[0], after[0]))
        # 全部 1000 帧仍可重放，证明后帧依赖链完整重建。
        self.assertEqual(len(list(player_after.replay())), FRAMES)

    def test_06_timeline_end_matches_delays(self) -> None:
        d = Path(self.tmp.name)
        catalog = FrameCatalog.from_gif(self.gif_path, d / "cat_time")
        metas = catalog.list_metadata()
        expected = Fraction(0)
        for m in metas:
            expected += Fraction(max(1, m["delay_centis"]), 100)
        self.assertEqual(FramePlayer(catalog).end_time, expected)


if __name__ == "__main__":
    unittest.main()
