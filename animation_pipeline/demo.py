"""端到端演示：生成真实 GIF，建立可编辑目录，生产数据集，并演示修订重发。

运行：.venv/bin/python -m animation_pipeline.demo
产物全部落在 var/demo/ 下。
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from .catalog import FrameCatalog
from .compositor import (
    FramePlayer,
    produce_dataset,
    reference_replay,
    rgba_digest,
)
from .generator import build_demo_gif

ROOT = Path(__file__).resolve().parent.parent / "var" / "demo"


def _check_reference(catalog: FrameCatalog) -> None:
    """纯 Python 二维数组参考实现 vs numpy 引擎逐像素核对。"""
    player = FramePlayer(catalog)
    ref = reference_replay(catalog)
    for i, display in enumerate(player.replay()):
        assert display.frame_id == ref[i]["frame_id"]
        if np.asarray(display.canvas).tolist() != ref[i]["canvas"]:
            raise AssertionError(f"帧 {i} 与参考实现逐像素不一致")
        assert display.overwritten_pixels == ref[i]["overwritten_pixels"]
        assert display.transparent_pixels == ref[i]["transparent_pixels"]
    print(f"  参考实现逐像素核对通过：{len(ref)} 帧")


def _check_png_samples(dataset_dir: Path) -> None:
    """重新读回 PNG，核对真实 RGBA 与 JSONL 摘要一致。"""
    with open(dataset_dir / "samples.jsonl", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    checked = 0
    for rec in records:
        if rec["ended"]:
            continue
        with Image.open(dataset_dir / rec["png"]) as im:
            arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        assert rgba_digest(arr) == rec["rgba_digest"], "PNG 回读摘要不匹配"
        assert arr.shape == (rec["canvas_height"], rec["canvas_width"], 4)
        checked += 1
    print(f"  PNG 回读验证通过：{checked} 个采样，结束样本 {sum(r['ended'] for r in records)} 个")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    gif_path = ROOT / "demo.gif"
    notes = build_demo_gif(str(gif_path))
    print(f"1) 已生成真实 GIF：{gif_path}（{len(notes['frames'])} 帧）")
    for i, note in enumerate(notes["frames"]):
        print(f"   帧{i}: {note}")

    catalog_v1_dir = ROOT / "catalog_v1"
    catalog = FrameCatalog.from_gif(gif_path, catalog_v1_dir)
    print(f"2) 目录产品 v1：{catalog_v1_dir}，revision={catalog.revision}，"
          f"画布 {catalog.width}x{catalog.height}，背景 {catalog.background_rgb}")
    _check_reference(catalog)

    # 有理数时间点：包含起点、帧边界（半开区间右沿）、0 延时帧、结束之后。
    time_points = [
        Fraction(0), Fraction(1, 100), Fraction(10, 100),
        Fraction(15, 100), Fraction(22, 100), Fraction(28, 100),
        Fraction(36, 100), Fraction(40, 100), Fraction(50, 100),
        Fraction(59, 100), Fraction(60, 100), Fraction(100, 1),
    ]
    dataset_v1_dir = ROOT / "dataset_v1"
    manifest = produce_dataset(catalog, dataset_v1_dir, time_points)
    print(f"3) 数据集 v1：{dataset_v1_dir}，样本 {manifest['sample_count']}，"
          f"唯一 PNG {manifest['unique_png_count']}，结束时间 {manifest['end_time_seconds']}s")
    _check_png_samples(dataset_v1_dir)

    # 编辑早期帧：把帧 0 的红底改成深蓝并移动帧 1 的矩形，再发布新产品。
    first_id = catalog.frame_ids[0]
    new_body = np.zeros((catalog.height, catalog.width, 4), dtype=np.uint8)
    new_body[:, :, :3] = (0, 40, 120)
    new_body[:, :, 3] = 255
    catalog.replace_body(first_id, new_body)
    second_id = catalog.frame_ids[1]
    second = catalog.get_frame(second_id)
    catalog.replace_rect_and_body(
        second_id, (0, 4, second.width, second.height), second.rgba.copy()
    )
    catalog_v2_dir = ROOT / "catalog_v2"
    catalog.publish_to(catalog_v2_dir)
    print(f"4) 修订早期帧后发布目录 v2：{catalog_v2_dir}，revision={catalog.revision}")
    _check_reference(catalog)

    dataset_v2_dir = ROOT / "dataset_v2"
    produce_dataset(catalog, dataset_v2_dir, time_points)
    print(f"5) 数据集 v2：{dataset_v2_dir}（后帧已按修订自动重算）")
    _check_png_samples(dataset_v2_dir)

    # 删除一帧验证 id 不复用，再新增。
    removed = catalog.frame_ids[2]
    existing_after_delete = list(catalog.frame_ids)
    catalog.delete_frame(removed)
    added = catalog.add_frame(
        rect=(0, 0, 2, 2),
        rgba=np.full((2, 2, 4), 255, dtype=np.uint8),
        delay_centis=3,
        disposal=1,
    )
    assert removed not in catalog.frame_ids
    assert added not in existing_after_delete and added > max(existing_after_delete)
    catalog.save()
    print(f"6) 删除 fid={removed} 后新增 fid={added}（身份不复用），"
          f"现存帧 {catalog.frame_ids}")
    print("演示完成。")


if __name__ == "__main__":
    main()
