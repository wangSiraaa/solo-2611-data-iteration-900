"""替片校核单：对指定展柜合同回答“替换片条能否插回”。

判定口径（全程不枚举行入场底画）：

* 两片有效总时长必须相同；
* 对柜方指定的每个样本时刻，以及（可选的）离场底画区域，装入*同一套合法整柜
  底画*时两片逐像素必须完全一致；
* 每个像素的颜色在片条端只有两种符号来源：入场延续（值=该像素所属饰面区的入场
  颜色）或确定颜色常量。两条片在该像素的符号项：
    - 入场/入场：恒等；
    - 常量/常量且 RGB 相同：恒等；
    - 其余组合只在*存在一套真实可装整柜方案*能暴露差异时才判失败，并把该方案
      连同最早不符的样本、坐标与两侧实际颜色写入校核单。
* 整柜根本配不齐时直接报告“柜约无解”，不能空判替片成功；
* 离场检查排在全部显示采样之后；核对方案不实际扣库存。

签单带片条指纹与柜约/库存指纹，片条换版、柜约或库存调整后可重开复读并重新判定。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .cabinet import (
    LegacyCabinet,
    PaintCanCabinet,
    RGB,
    Scheme,
)
from .filmstrip import ENTRY_TERM, ENTRY_TERM_ID, Filmstrip, Term
from .compositor import normalize_time

SHEET_FORMAT = "animgif-substitution-sheet"
SHEET_VERSION = 1

STAGE_DISPLAY = "display"
STAGE_EXIT = "exit"


class SubstitutionInputError(ValueError):
    """两片时长不一致、时间点非法等输入问题。"""


# --------------------------------------------------------------------------- #
# 报告数据结构
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Mismatch:
    stage: str
    sample_id: str | None
    time_fraction: str | None
    coord: tuple[int, int]
    region: str
    rgb_a: RGB
    rgb_b: RGB
    term_a: dict
    term_b: dict
    reason_text: str


@dataclass(frozen=True)
class SubstitutionReport:
    success: bool
    verdict: str  # "interchangeable" | "rejected" | "cabinet_unsatisfiable" | "input_error"
    detail: str
    strip_a_id: str
    strip_b_id: str
    duration_a: str
    duration_b: str
    checked_samples: int
    checked_exit_regions: tuple[str, ...]
    earliest_mismatch: Mismatch | None = None
    witness_scheme: dict | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "verdict": self.verdict,
            "detail": self.detail,
            "strip_a_id": self.strip_a_id,
            "strip_b_id": self.strip_b_id,
            "duration_seconds": {"a": self.duration_a, "b": self.duration_b},
            "checked_samples": self.checked_samples,
            "checked_exit_regions": list(self.checked_exit_regions),
            "earliest_mismatch": (
                _mismatch_to_dict(self.earliest_mismatch)
                if self.earliest_mismatch is not None
                else None
            ),
            "witness_scheme": self.witness_scheme,
        }


def _mismatch_to_dict(m: Mismatch) -> dict:
    return {
        "stage": m.stage,
        "sample_id": m.sample_id,
        "time_fraction": m.time_fraction,
        "coord": list(m.coord),
        "region": m.region,
        "rgb_a": list(m.rgb_a),
        "rgb_b": list(m.rgb_b),
        "term_a": m.term_a,
        "term_b": m.term_b,
        "reason": m.reason_text,
    }


def _scheme_to_dict(scheme: Scheme, cabinet) -> dict:
    paints = [
        {"region": p.region, "lot_id": p.lot_id, "rgb": list(p.rgb)}
        for p in scheme.paints
    ]
    return {
        "cabinet_kind": "paint_can" if isinstance(cabinet, PaintCanCabinet) else "legacy",
        "colors": {name: list(rgb) for name, rgb in scheme.colors.items()},
        "paints": paints,
    }


# --------------------------------------------------------------------------- #
# 柜约指纹：重开签单时识别柜约/库存变化（不按颜色并库，批号逐列入指纹）。
# --------------------------------------------------------------------------- #

def cabinet_fingerprint(cabinet) -> str:
    if isinstance(cabinet, LegacyCabinet):
        payload = {
            "kind": "legacy",
            "width": cabinet.width,
            "height": cabinet.height,
            "regions": [
                {"name": r.name, "rect": [r.left, r.top, r.width, r.height]}
                for r in cabinet.regions
            ],
            "color_lists": {
                name: [list(c) for c in colors]
                for name, colors in cabinet.color_lists.items()
            },
        }
    elif isinstance(cabinet, PaintCanCabinet):
        payload = {
            "kind": "paint_can",
            "width": cabinet.width,
            "height": cabinet.height,
            "regions": [
                {"name": r.name, "rect": [r.left, r.top, r.width, r.height]}
                for r in cabinet.regions
            ],
            "lots": [
                {"lot_id": lot.lot_id, "rgb": list(lot.rgb), "cans": lot.cans}
                for lot in cabinet.lots.values()
            ],
            "compatible_lots": {
                name: list(ids) for name, ids in cabinet.compatible_lots.items()
            },
        }
    else:  # pragma: no cover
        raise SubstitutionInputError("未知展柜类型")
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# 入场底画
# --------------------------------------------------------------------------- #

def entry_canvas_from_scheme(cabinet, scheme: Scheme) -> np.ndarray:
    """把整柜方案逐区填满成 uint8 RGBA 入场底画。"""
    canvas = np.zeros((cabinet.height, cabinet.width, 4), dtype=np.uint8)
    for r in cabinet.regions:
        rgb = scheme.region_color(r.name)
        canvas[r.top : r.top + r.height, r.left : r.left + r.width, :3] = rgb
    canvas[:, :, 3] = 255
    return canvas


# --------------------------------------------------------------------------- #
# 符号项比较与反例方案
# --------------------------------------------------------------------------- #

def _terms_always_equal(term_a: Term, term_b: Term) -> bool:
    if term_a.is_entry and term_b.is_entry:
        return True
    if not term_a.is_entry and not term_b.is_entry:
        return term_a.rgb == term_b.rgb
    return False


def _allowed_region_colors(cabinet, region_name: str) -> list[RGB]:
    if isinstance(cabinet, LegacyCabinet):
        return list(cabinet.color_lists[region_name])
    # 新柜：颜色只能由兼容该区的批号刷出；同色批号分别保留，但候选色去重即可。
    seen: dict[RGB, None] = {}
    for lot_id in cabinet.compatible_lots[region_name]:
        seen.setdefault(cabinet.lots[lot_id].rgb, None)
    return list(seen)


def _witness_scheme(cabinet, region_name: str, term_a: Term, term_b: Term) -> Scheme | None:
    """求一套真实可装、且能在该像素暴露两片差异的整柜方案；不存在返回 None。"""
    if _terms_always_equal(term_a, term_b):
        return None
    if not term_a.is_entry and not term_b.is_entry:
        # 两个不同的确定颜色：任何可装方案都暴露差异。
        return cabinet.any_scheme()
    # 一侧入场、一侧常量：需要该区入场色 != 常量色。
    const_rgb = term_b.rgb if term_a.is_entry else term_a.rgb
    for rgb in _allowed_region_colors(cabinet, region_name):
        if rgb == const_rgb:
            continue
        scheme = cabinet.scheme_with_region_color(region_name, rgb)
        if scheme is not None:
            return scheme
    return None


def _term_of(trace, term_id: int) -> Term:
    if term_id == ENTRY_TERM_ID:
        return ENTRY_TERM
    return trace.terms[term_id]


def _evaluate(term: Term, entry: np.ndarray, x: int, y: int) -> RGB:
    if term.is_entry:
        return tuple(int(c) for c in entry[y, x, :3])  # type: ignore[return-value]
    assert term.rgb is not None
    return term.rgb


# --------------------------------------------------------------------------- #
# 主入口：出具替片校核单
# --------------------------------------------------------------------------- #

@dataclass
class CheckRequest:
    sample_id: str
    time_point: object  # int/Fraction/float/分数字符串


def verify_substitution(
    strip_a: Filmstrip,
    strip_b: Filmstrip,
    cabinet,
    sample_requests: list[CheckRequest],
    *,
    exit_regions: list[str] | None = None,
) -> SubstitutionReport:
    exit_names = tuple(exit_regions or ())
    unknown = [name for name in exit_names if name not in {r.name for r in cabinet.regions}]
    if unknown:
        raise SubstitutionInputError(f"离场检查引用了未知饰面区：{unknown}")
    if (strip_a.width, strip_a.height) != (cabinet.width, cabinet.height):
        raise SubstitutionInputError("片条逻辑画布与展柜画布尺寸不一致")
    if (strip_b.width, strip_b.height) != (cabinet.width, cabinet.height):
        raise SubstitutionInputError("片条逻辑画布与展柜画布尺寸不一致")

    duration_a, duration_b = strip_a.duration, strip_b.duration
    base = SubstitutionReport(
        success=False,
        verdict="input_error",
        detail="",
        strip_a_id=strip_a.strip_id,
        strip_b_id=strip_b.strip_id,
        duration_a=str(duration_a),
        duration_b=str(duration_b),
        checked_samples=len(sample_requests),
        checked_exit_regions=exit_names,
    )
    if duration_a != duration_b:
        return SubstitutionReport(
            success=False,
            verdict="input_error",
            detail=f"两片有效总时长不同：{duration_a} != {duration_b}，不得比较",
            strip_a_id=strip_a.strip_id,
            strip_b_id=strip_b.strip_id,
            duration_a=str(duration_a),
            duration_b=str(duration_b),
            checked_samples=len(sample_requests),
            checked_exit_regions=exit_names,
        )

    # 整柜配不齐：柜约无解，禁止空判成功。
    if not cabinet.is_satisfiable():
        return SubstitutionReport(
            success=False,
            verdict="cabinet_unsatisfiable",
            detail="整柜没有任何可同时配齐的底画方案：柜约无解，不能判替片成功",
            strip_a_id=strip_a.strip_id,
            strip_b_id=strip_b.strip_id,
            duration_a=str(duration_a),
            duration_b=str(duration_b),
            checked_samples=len(sample_requests),
            checked_exit_regions=exit_names,
        )

    trace_a = strip_a.symbolic_trace()
    trace_b = strip_b.symbolic_trace()
    requests = [
        CheckRequest(req.sample_id, normalize_time(req.time_point))
        for req in sample_requests
    ]

    # -- 显示采样：按样本顺序，样本内按行主序坐标找最早不符 -------------- #
    for req in requests:
        idx_a = strip_a.select_index(req.time_point)
        idx_b = strip_b.select_index(req.time_point)
        if idx_a is None or idx_b is None:
            raise SubstitutionInputError(
                f"样本 {req.sample_id} 落在片条结束之后（t={req.time_point}）"
            )
        grid_a = trace_a.display_grids[idx_a]
        grid_b = trace_b.display_grids[idx_b]
        mismatch = _scan_grid(
            cabinet,
            strip_a,
            strip_b,
            grid_a,
            grid_b,
            trace_a,
            trace_b,
            stage=STAGE_DISPLAY,
            sample_id=req.sample_id,
            time_point=req.time_point,
        )
        if mismatch is not None:
            return _reject_report(base, mismatch, cabinet)

    # -- 离场底画：排在全部采样之后，只查指定区 --------------------------- #
    if exit_names:
        mismatch = _scan_grid(
            cabinet,
            strip_a,
            strip_b,
            trace_a.exit_grid,
            trace_b.exit_grid,
            trace_a,
            trace_b,
            stage=STAGE_EXIT,
            sample_id=None,
            time_point=None,
            only_regions=set(exit_names),
        )
        if mismatch is not None:
            return _reject_report(base, mismatch, cabinet)

    return SubstitutionReport(
        success=True,
        verdict="interchangeable",
        detail=(
            f"两片在 {len(requests)} 个显示采样"
            + (f"与 {len(exit_names)} 个离场区" if exit_names else "")
            + "上，对一切合法整柜底画逐像素一致"
        ),
        strip_a_id=strip_a.strip_id,
        strip_b_id=strip_b.strip_id,
        duration_a=str(duration_a),
        duration_b=str(duration_b),
        checked_samples=len(requests),
        checked_exit_regions=exit_names,
    )


def _scan_grid(
    cabinet,
    strip_a: Filmstrip,
    strip_b: Filmstrip,
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    trace_a,
    trace_b,
    *,
    stage: str,
    sample_id: str | None,
    time_point: Fraction | None,
    only_regions: set[str] | None = None,
) -> tuple[Mismatch, Scheme] | None:
    h, w = grid_a.shape
    for y in range(h):
        for x in range(w):
            region_name = cabinet.region_at(x, y)
            if only_regions is not None and region_name not in only_regions:
                continue
            term_a = _term_of(trace_a, int(grid_a[y, x]))
            term_b = _term_of(trace_b, int(grid_b[y, x]))
            if _terms_always_equal(term_a, term_b):
                continue
            scheme = _witness_scheme(cabinet, region_name, term_a, term_b)
            if scheme is None:
                # 该区在所有合法方案下都无法暴露差异，按全域量化不算失败。
                continue
            entry = entry_canvas_from_scheme(cabinet, scheme)
            rgb_a = _evaluate(term_a, entry, x, y)
            rgb_b = _evaluate(term_b, entry, x, y)
            reason_text = _reason_text(stage, term_a, term_b, region_name, scheme)
            mismatch = Mismatch(
                stage=stage,
                sample_id=sample_id,
                time_fraction=str(time_point) if time_point is not None else None,
                coord=(x, y),
                region=region_name,
                rgb_a=rgb_a,
                rgb_b=rgb_b,
                term_a=term_a.to_dict(),
                term_b=term_b.to_dict(),
                reason_text=reason_text,
            )
            return mismatch, scheme
    return None


def _reject_report(
    base: SubstitutionReport,
    found: tuple[Mismatch, Scheme],
    cabinet,
) -> SubstitutionReport:
    mismatch, scheme = found
    if mismatch.stage == STAGE_DISPLAY:
        detail = (
            f"替片被拒：样本 {mismatch.sample_id}（t={mismatch.time_fraction}）"
            f"坐标 {mismatch.coord}（区 {mismatch.region}）两侧颜色不同"
        )
    else:
        detail = (
            f"替片被拒：离场底画在区 {mismatch.region} 坐标 {mismatch.coord} 颜色不同，"
            "无法备下一片叠印"
        )
    return SubstitutionReport(
        success=False,
        verdict="rejected",
        detail=detail,
        strip_a_id=base.strip_a_id,
        strip_b_id=base.strip_b_id,
        duration_a=base.duration_a,
        duration_b=base.duration_b,
        checked_samples=base.checked_samples,
        checked_exit_regions=base.checked_exit_regions,
        earliest_mismatch=mismatch,
        witness_scheme=_scheme_to_dict(scheme, cabinet),
    )


def _reason_text(
    stage: str, term_a: Term, term_b: Term, region_name: str, scheme: Scheme
) -> str:
    parts = []
    for label, term in (("甲片", term_a), ("乙片", term_b)):
        if term.is_entry:
            parts.append(
                f"{label}该像素是入场延续，取区 {region_name} 入场色 "
                f"{scheme.region_color(region_name)}"
            )
        else:
            parts.append(
                f"{label}该像素由帧{term.frame_id}的{term.reason}提供确定色 {term.rgb}"
            )
    prefix = "显示不符：" if stage == STAGE_DISPLAY else "离场不符："
    return prefix + "；".join(parts)


# --------------------------------------------------------------------------- #
# 签单持久化与重开
# --------------------------------------------------------------------------- #

def issue_sheet(
    output_path: str | os.PathLike,  # type: ignore[name-defined]
    strip_a: Filmstrip,
    strip_b: Filmstrip,
    cabinet,
    sample_requests: list[CheckRequest],
    *,
    exit_regions: list[str] | None = None,
) -> SubstitutionReport:
    report = verify_substitution(
        strip_a, strip_b, cabinet, sample_requests, exit_regions=exit_regions
    )
    payload = {
        "format": SHEET_FORMAT,
        "version": SHEET_VERSION,
        "report": report.to_dict(),
        "fingerprints": {
            "strip_a": strip_a.fingerprint(),
            "strip_b": strip_b.fingerprint(),
            "cabinet": cabinet_fingerprint(cabinet),
        },
        "strip_versions": {
            "strip_a": strip_a.strip_version,
            "strip_b": strip_b.strip_version,
        },
        "request": {
            "samples": [
                {"sample_id": req.sample_id, "time_fraction": str(normalize_time(req.time_point))}
                for req in sample_requests
            ],
            "exit_regions": list(exit_regions or ()),
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return report


def reopen_sheet(
    sheet_path: str | os.PathLike,  # type: ignore[name-defined]
    strip_a: Filmstrip,
    strip_b: Filmstrip,
    cabinet,
) -> dict:
    """重开先前签单：复读旧结论，核对指纹并按当前片条/柜约/库存重新判定。"""
    path = Path(sheet_path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("format") != SHEET_FORMAT:
        raise SubstitutionInputError("不是本服务出具的替片校核单")

    current_fps = {
        "strip_a": strip_a.fingerprint(),
        "strip_b": strip_b.fingerprint(),
        "cabinet": cabinet_fingerprint(cabinet),
    }
    drift = {
        key: {
            "signed": payload["fingerprints"][key],
            "current": current_fps[key],
            "changed": payload["fingerprints"][key] != current_fps[key],
        }
        for key in current_fps
    }
    requests = [
        CheckRequest(item["sample_id"], Fraction(item["time_fraction"]))
        for item in payload["request"]["samples"]
    ]
    current_report = verify_substitution(
        strip_a,
        strip_b,
        cabinet,
        requests,
        exit_regions=payload["request"]["exit_regions"],
    )
    return {
        "sheet_path": str(path),
        "signed_verdict": payload["report"]["verdict"],
        "signed_detail": payload["report"]["detail"],
        "drift": drift,
        "anything_changed": any(item["changed"] for item in drift.values()),
        "current_report": current_report.to_dict(),
        "verdict_still_holds": current_report.verdict == payload["report"]["verdict"],
    }
