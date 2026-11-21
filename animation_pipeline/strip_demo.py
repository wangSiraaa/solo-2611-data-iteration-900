"""片条回放与替片校核端到端演示（题面全部场景）。

运行：.venv/bin/python -m animation_pipeline.strip_demo
产物落在 var/strip_demo/ 下：片条产品、片条样本、替片校核单 JSON。
"""

from __future__ import annotations

import json
from fractions import Fraction

from pathlib import Path

import numpy as np

from .cabinet import (
    LegacyCabinet,
    PaintCanCabinet,
    PaintLot,
    Region,
)
from .filmstrip import Filmstrip, produce_strip_samples
from .stripbuilders import (
    R,
    W,
    build_local_palette_strip,
    build_red_pair,
    build_white_pair,
    build_xy_pair,
    revise_local_palette_to_red,
)
from .substitution import (
    CheckRequest,
    entry_canvas_from_scheme,
    issue_sheet,
    reopen_sheet,
    verify_substitution,
)
from .catalog import FrameCatalog

ROOT = Path(__file__).resolve().parent.parent / "var" / "strip_demo"
T = Fraction(5, 100)


def _print_report(title: str, report) -> None:
    print(f"\n=== {title} ===")
    print(f"结论: {report.verdict}（success={report.success}）—— {report.detail}")
    if report.earliest_mismatch is not None:
        m = report.earliest_mismatch
        print(
            f"  最早不符: stage={m.stage} sample={m.sample_id} coord={m.coord} "
            f"区={m.region} 甲={m.rgb_a} 乙={m.rgb_b}"
        )
        print(f"  原因: {m.reason_text}")
    if report.witness_scheme is not None:
        print(f"  真实可装反例整柜方案: {json.dumps(report.witness_scheme, ensure_ascii=False)}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    req = [CheckRequest("s0", T)]

    # -- 场景 1：白逻辑背景，透明一拍 vs 画白保留 ------------------------ #
    print("########## 场景 1：白/红饰面下的透明片与画白片 ##########")
    strip_a, strip_b = build_white_pair(1)
    white_cab = LegacyCabinet(1, 1, [Region("P", 0, 0, 1, 1)], {"P": [W]})
    white_red_cab = LegacyCabinet(1, 1, [Region("P", 0, 0, 1, 1)], {"P": [W, R]})
    _print_report("白色饰面（允许替片）", verify_substitution(strip_a, strip_b, white_cab, req))
    _print_report(
        "增加红色饰面（拒绝并复现红/白差异）",
        verify_substitution(strip_a, strip_b, white_red_cab, req),
    )

    # 任意合法入场底画回放 + 来源追溯：同一透明片，白/红入场结果不同。
    samples_dir = ROOT / "strip_samples_red_entry"
    red_entry = np.zeros((1, 1, 4), dtype=np.uint8)
    red_entry[:, :, :3] = R
    red_entry[:, :, 3] = 255
    produce_strip_samples(strip_a, samples_dir, red_entry, [T, Fraction(1)])
    explain = strip_a.explain_pixel((0, 0), time_point=T)
    print(f"\n透明片红底入场像素来源: {json.dumps(explain, ensure_ascii=False)}")

    # -- 场景 2：均显示红，处置3 vs 处置1 -------------------------------- #
    print("\n########## 场景 2：均显示红，处置3 vs 处置1 ##########")
    strip_d3, strip_d1 = build_red_pair(1)
    _print_report(
        "只验显示（可换）",
        verify_substitution(strip_d3, strip_d1, white_cab, req),
    )
    _print_report(
        "要求离场底画同色（不能换）",
        verify_substitution(strip_d3, strip_d1, white_cab, req, exit_regions=["P"]),
    )

    # -- 场景 3：局部色表改稿与旧签单复读 -------------------------------- #
    print("\n########## 场景 3：局部色表改稿 + 旧签单复读 ##########")
    cat_dir = ROOT / "palette_catalog"
    catalog, pal_strip = build_local_palette_strip(1, product_dir=cat_dir)
    catalog.save()
    sheet_path = ROOT / "sheet_palette.json"
    # 白饰面柜上签单：改稿前后透明片都可与该片互换；改稿后再到白/红柜即被拒。
    issue_sheet(sheet_path, strip_a, pal_strip, white_cab, req)
    revise_local_palette_to_red(catalog, catalog.frame_ids[0])
    catalog.save()
    revised = Filmstrip.from_catalog(
        FrameCatalog.load(cat_dir), catalog.frame_ids, "local-palette-revision"
    )
    reopened = reopen_sheet(sheet_path, strip_a, revised, white_cab)
    print(
        f"旧签单 verdict={reopened['signed_verdict']}；"
        f"片条指纹漂移={reopened['drift']['strip_b']['changed']}；"
        f"白柜重判 verdict={reopened['current_report']['verdict']}"
        "（透明片显白、改红片显红，旧结论被改判而非沿用）"
    )
    red_after = verify_substitution(strip_a, revised, white_red_cab, req)
    print(f"同片放到白/红柜重判 verdict={red_after.verdict}")

    # -- 场景 4：两区 X/Y 柜，漆罐批号与库存 ----------------------------- #
    print("\n########## 场景 4：两区柜 X（白/红）、Y（仅红） ##########")
    regions = [Region("X", 0, 0, 1, 1), Region("Y", 1, 0, 1, 1)]
    xy_a, xy_b = build_xy_pair(2, 1)

    def can_cabinet(r_cans: int) -> PaintCanCabinet:
        return PaintCanCabinet(
            2,
            1,
            regions,
            [PaintLot("W1", W, 1), PaintLot("R1", R, r_cans)],
            {"X": ["W1", "R1"], "Y": ["R1"]},
        )

    _print_report("R 仅 1 罐（留给 Y，X 只能白 -> 可换）",
                  verify_substitution(xy_a, xy_b, can_cabinet(1), req))
    rejected = verify_substitution(xy_a, xy_b, can_cabinet(2), req)
    _print_report("R 增至 2 罐（X 可红 -> 不能换）", rejected)
    # 用反例方案真实回放，还原 X 上的红/白差异。
    from .cabinet import RegionPaint, Scheme

    witness = Scheme(
        colors={"X": R, "Y": R},
        paints=(RegionPaint("X", "R1", R), RegionPaint("Y", "R1", R)),
    )
    cabinet2 = can_cabinet(2)
    assert cabinet2.supports_scheme(witness)
    entry = entry_canvas_from_scheme(cabinet2, witness)
    ax = tuple(int(c) for c in xy_a.exit_canvas(entry)[0, 0, :3])
    bx = tuple(int(c) for c in xy_b.exit_canvas(entry)[0, 0, :3])
    print(f"  反例真实回放 X 列：甲={ax} 乙={bx}（红/白差异复现）")

    _print_report("R 缺货（整柜无解，禁止空判成功）",
                  verify_substitution(xy_a, xy_b, can_cabinet(0), req))

    # -- 场景 5：同 RGB 两批号各自只适配部分区，不能并库 ----------------- #
    print("\n########## 场景 5：同色批号部分适配，禁止按颜色并库 ##########")
    split = PaintCanCabinet(
        2,
        1,
        regions,
        [PaintLot("RA", R, 1), PaintLot("RB", R, 1)],
        {"X": ["RA"], "Y": ["RB"]},
    )
    _print_report("RA 供 X、RB 供 Y（批号分别计库存）",
                  verify_substitution(xy_a, xy_b, split, req))
    stockout = PaintCanCabinet(
        2,
        1,
        regions,
        [PaintLot("RA", R, 0), PaintLot("RB", R, 1)],
        {"X": ["RA"], "Y": ["RB"]},
    )
    _print_report("RA 缺货（并库会误判，正确判柜约无解）",
                  verify_substitution(xy_a, xy_b, stockout, req))

    # 落盘一份正式校核单，演示可重开。
    issue_sheet(ROOT / "sheet_xy_two_red.json", xy_a, xy_b, can_cabinet(2), req)
    print(f"\n演示产物目录: {ROOT}")
    print("演示完成。")


if __name__ == "__main__":
    main()
