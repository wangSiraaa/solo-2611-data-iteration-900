"""可重用片条产品与任意入场底画回放。

片条是从帧目录切出的*固定产品*：记录按原顺序的连续帧身份、原逻辑画布尺寸与
该片目录的逻辑背景色。它与“整片播放器”（compositor.FramePlayer）的关键区别：

* 片条不继承首帧之前的任何电影内容——进入片条时，每个像素携带的是*调用方整张
  入场底画*的颜色；
* 处置 2 的背景恢复一律取片条自己的逻辑背景色，不把它当成调用方入场色；
* 片条端可做*符号回放*：不枚举行任何入场底画，直接给出任一显示时刻与离场底画上
  每个像素的来源项——要么是某坐标入场像素的延续，要么是由确定 RGB 提供的常量项
  （帧正文覆盖 / 处置2 背景恢复），并带可定位到帧的原因。

合成口径与 compositor 完全一致（处置 0 按 1、透明不覆盖、先存显示再处置、
处置 3 恢复绘制前画布），只是初始画布换成调用方入场底画。
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from .catalog import FrameCatalog, RawFrame
from .compositor import CENTIS, effective_delay_centis, normalize_time

STRIP_FORMAT = "animgif-filmstrip"
STRIP_VERSION = 1

# 符号项 id：0 保留给“入场像素延续”，正整数为确定颜色常量项。
ENTRY_TERM_ID = 0

REASON_PAINT = "paint"                       # 帧正文不透明覆盖（颜色已含局部色表解析）
REASON_RESTORE_BACKGROUND = "restore_background"  # 处置 2：恢复片条逻辑背景
REASON_TRANSPARENT = "transparent"           # 透明像素：不覆盖，沿用画前内容
REASON_RESTORE_BEFORE = "restore_before"     # 处置 3：恢复绘制前内容


class FilmstripValidationError(ValueError):
    """片条构造/入场底画/时间点非法时抛出。"""


@dataclass(frozen=True)
class Term:
    """一个像素的符号来源项。"""

    kind: str  # "entry" | "constant"
    rgb: tuple[int, int, int] | None = None
    frame_id: int | None = None
    reason: str | None = None
    detail: str | None = None

    @property
    def is_entry(self) -> bool:
        return self.kind == "entry"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "rgb": list(self.rgb) if self.rgb is not None else None,
            "frame_id": self.frame_id,
            "reason": self.reason,
            "detail": self.detail,
        }


ENTRY_TERM = Term(kind="entry", detail="入场底画同一坐标像素的延续")


@dataclass(frozen=True)
class StripDisplayFrame:
    frame_id: int
    start: Fraction
    end: Fraction
    canvas: np.ndarray
    overwritten_pixels: int
    transparent_pixels: int


@dataclass
class SymbolicTrace:
    """一次符号回放的结果，与入场底画无关，可缓存复用。

    display_grids[i] 为第 i 帧绘制完成后的符号项 id 网格；
    exit_grid 为全部处置结束后的离场符号项网格。
    """

    frame_ids: list[int]
    display_grids: list[np.ndarray]
    exit_grid: np.ndarray
    terms: dict[int, Term]
    starts: list[Fraction]
    ends: list[Fraction]

    def term_at(self, term_id: int) -> Term:
        if term_id == ENTRY_TERM_ID:
            return ENTRY_TERM
        return self.terms[term_id]

    def display_grid_at(self, index: int) -> np.ndarray:
        return self.display_grids[index]


class Filmstrip:
    """切出的片条产品。可纯内存使用，也可绑定目录持久化。"""

    def __init__(
        self,
        strip_id: str,
        width: int,
        height: int,
        background_rgb: tuple[int, int, int],
        frames: list[RawFrame],
        *,
        strip_version: int = 1,
        product_dir: str | os.PathLike[str] | None = None,
        source_catalog: str | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise FilmstripValidationError("画布宽高必须为正整数")
        if not frames:
            raise FilmstripValidationError("片条至少要包含一帧")
        if len(background_rgb) != 3 or any(not 0 <= c <= 255 for c in background_rgb):
            raise FilmstripValidationError("背景 RGB 必须是 3 个 0..255 整数")
        for f in frames:
            if f.left < 0 or f.top < 0 or f.left + f.width > width or f.top + f.height > height:
                raise FilmstripValidationError(
                    f"帧 {f.frame_id} 矩形越界（画布 {width}x{height}）"
                )
        self.strip_id = str(strip_id)
        self.strip_version = int(strip_version)
        self.width = int(width)
        self.height = int(height)
        self.background_rgb = tuple(int(c) for c in background_rgb)
        self._frames: list[RawFrame] = list(frames)
        self.product_dir = Path(product_dir) if product_dir is not None else None
        self.source_catalog = source_catalog
        self._trace: SymbolicTrace | None = None
        self._trace_fingerprint: str | None = None
        if self.product_dir is not None:
            (self.product_dir / "bodies").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 从帧目录切出：必须是按目录顺序的连续帧身份；不继承首帧之前的内容。
    # ------------------------------------------------------------------ #
    @classmethod
    def from_catalog(
        cls,
        catalog: FrameCatalog,
        frame_ids: list[int] | tuple[int, ...],
        strip_id: str,
        *,
        product_dir: str | os.PathLike[str] | None = None,
        strip_version: int = 1,
    ) -> "Filmstrip":
        ordered_ids = catalog.frame_ids
        pos = {fid: i for i, fid in enumerate(ordered_ids)}
        ids = list(frame_ids)
        if not ids:
            raise FilmstripValidationError("切出片条至少需要一帧")
        if any(fid not in pos for fid in ids):
            raise FilmstripValidationError("存在不属于该目录的 frame_id")
        positions = [pos[fid] for fid in ids]
        if positions != list(range(min(positions), max(positions) + 1)):
            raise FilmstripValidationError("片条只接受按原顺序的连续帧身份")
        frames = [catalog.get_frame(fid) for fid in ids]
        source = str(catalog.product_dir) if catalog.product_dir is not None else None
        strip = cls(
            strip_id,
            catalog.width,
            catalog.height,
            catalog.background_rgb,
            frames,
            strip_version=strip_version,
            product_dir=product_dir,
            source_catalog=source,
        )
        if product_dir is not None:
            strip.save()
        return strip

    # ------------------------------------------------------------------ #
    # 基本属性
    # ------------------------------------------------------------------ #
    @property
    def frame_ids(self) -> list[int]:
        return [f.frame_id for f in self._frames]

    def frame_count(self) -> int:
        return len(self._frames)

    def iter_frames(self) -> list[RawFrame]:
        return list(self._frames)

    @property
    def duration(self) -> Fraction:
        """有效总时长（秒）：0 延时按 1 处理。"""
        total = sum(effective_delay_centis(f.delay_centis) for f in self._frames)
        return total * CENTIS

    def _timeline(self) -> tuple[list[Fraction], list[Fraction]]:
        starts: list[Fraction] = []
        ends: list[Fraction] = []
        t = Fraction(0)
        for f in self._frames:
            starts.append(t)
            t += effective_delay_centis(f.delay_centis) * CENTIS
            ends.append(t)
        return starts, ends

    def select_index(self, time_point) -> int | None:
        """与整片播放器相同的半开区间选帧；结束之后返回 None。"""
        t = normalize_time(time_point)
        starts, ends = self._timeline()
        if t >= ends[-1]:
            return None
        idx = bisect.bisect_right(starts, t) - 1
        return max(idx, 0)

    # ------------------------------------------------------------------ #
    # 指纹：签单据此识别片条换版。
    # ------------------------------------------------------------------ #
    def fingerprint(self) -> str:
        payload = {
            "strip_id": self.strip_id,
            "strip_version": self.strip_version,
            "width": self.width,
            "height": self.height,
            "background_rgb": list(self.background_rgb),
            "frames": [
                {
                    "frame_id": f.frame_id,
                    "rect": [f.left, f.top, f.width, f.height],
                    "delay_centis": f.delay_centis,
                    "disposal": f.disposal,
                    "palette": [list(c) for c in f.palette] if f.palette is not None else None,
                    "palette_is_local": f.palette_is_local,
                    "transparent_index": f.transparent_index,
                    "body_sha256": hashlib.sha256(
                        np.ascontiguousarray(f.rgba).tobytes()
                    ).hexdigest(),
                }
                for f in self._frames
            ],
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ------------------------------------------------------------------ #
    # 入场底画校验
    # ------------------------------------------------------------------ #
    def _coerce_entry(self, entry_rgba: np.ndarray) -> np.ndarray:
        arr = np.asarray(entry_rgba)
        if arr.dtype != np.uint8 or arr.ndim not in (2, 3):
            raise FilmstripValidationError("入场底画必须是 uint8 数组")
        if arr.ndim == 2:
            raise FilmstripValidationError("入场底画需要 RGB(A) 通道")
        if arr.shape[2] == 3:
            alpha = np.full((self.height, self.width, 1), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=2).astype(np.uint8)
        if arr.shape != (self.height, self.width, 4):
            raise FilmstripValidationError(
                f"入场底画形状 {arr.shape} 与片条画布 "
                f"{(self.height, self.width, 4)} 不符"
            )
        return arr.copy()

    # ------------------------------------------------------------------ #
    # 具体回放：给定调用方整张入场底画，逐帧产出显示画布与离场底画。
    # ------------------------------------------------------------------ #
    def replay(self, entry_rgba: np.ndarray):
        canvas = self._coerce_entry(entry_rgba)
        starts, ends = self._timeline()
        for i, frame in enumerate(self._frames):
            before = canvas.copy() if frame.disposal == 3 else None
            overwritten, transparent = self._draw(canvas, frame)
            yield StripDisplayFrame(
                frame_id=frame.frame_id,
                start=starts[i],
                end=ends[i],
                canvas=canvas.copy(),
                overwritten_pixels=overwritten,
                transparent_pixels=transparent,
            )
            disposal = frame.disposal if frame.disposal != 0 else 1
            if disposal == 2:
                self._restore_background(canvas, frame)
            elif disposal == 3:
                canvas = before  # type: ignore[assignment]

    def exit_canvas(self, entry_rgba: np.ndarray) -> np.ndarray:
        """全部处置之后留给下一片的离场底画。"""
        canvas = self._coerce_entry(entry_rgba)
        for frame in self._frames:
            before = canvas.copy() if frame.disposal == 3 else None
            self._draw(canvas, frame)
            disposal = frame.disposal if frame.disposal != 0 else 1
            if disposal == 2:
                self._restore_background(canvas, frame)
            elif disposal == 3:
                canvas = before  # type: ignore[assignment]
        return canvas

    def display_at_time(self, entry_rgba: np.ndarray, time_point):
        t = normalize_time(time_point)
        index = self.select_index(t)
        if index is None:
            return None
        for i, display in enumerate(self.replay(entry_rgba)):
            if i == index:
                return display
        return None  # pragma: no cover

    def _draw(self, canvas: np.ndarray, frame: RawFrame) -> tuple[int, int]:
        region = canvas[
            frame.top : frame.top + frame.height,
            frame.left : frame.left + frame.width,
        ]
        mask = frame.rgba[:, :, 3] > 0
        region[mask] = frame.rgba[mask]
        return int(mask.sum()), int((~mask).sum())

    def _restore_background(self, canvas: np.ndarray, frame: RawFrame) -> None:
        """处置 2：恢复*片条目录*的逻辑背景，与调用方入场色无关。"""
        region = canvas[
            frame.top : frame.top + frame.height,
            frame.left : frame.left + frame.width,
        ]
        region[:, :, :3] = np.asarray(self.background_rgb, dtype=np.uint8)
        region[:, :, 3] = 255

    # ------------------------------------------------------------------ #
    # 符号回放：每个像素只区分“入场延续”与“确定颜色常量项”。
    # 与入场底画无关，整片只算一遍，校核任意多套合法入场底画都复用它。
    # ------------------------------------------------------------------ #
    def symbolic_trace(self) -> SymbolicTrace:
        fp = self.fingerprint()
        if self._trace is not None and self._trace_fingerprint == fp:
            return self._trace

        state = np.zeros((self.height, self.width), dtype=np.int32)
        terms: dict[int, Term] = {}
        next_id = 1
        const_cache: dict[tuple, int] = {}

        def const_id(rgb: tuple[int, int, int], frame_id: int, reason: str, detail: str) -> int:
            nonlocal next_id
            key = (rgb, frame_id, reason)
            if key not in const_cache:
                const_cache[key] = next_id
                terms[next_id] = Term(
                    kind="constant", rgb=rgb, frame_id=frame_id, reason=reason, detail=detail
                )
                next_id += 1
            return const_cache[key]

        display_grids: list[np.ndarray] = []
        starts, ends = self._timeline()
        bg = tuple(int(c) for c in self.background_rgb)

        for frame in self._frames:
            if frame.disposal == 3:
                before = state.copy()
            else:
                before = None

            # 不透明覆盖：颜色即正文 RGBA（局部色表已在目录层解析进 RGB）。
            opaque = frame.rgba[:, :, 3] > 0
            if opaque.any():
                palette_note = "局部色表" if frame.palette_is_local else "帧正文"
                detail = f"帧{frame.frame_id}不透明覆盖（{palette_note}确定颜色）"
                color_to_id: dict[tuple[int, int, int], int] = {}
                ys, xs = np.nonzero(opaque)
                for dy, dx in zip(ys, xs):
                    rgb = tuple(int(c) for c in frame.rgba[dy, dx, :3])
                    if rgb not in color_to_id:
                        color_to_id[rgb] = const_id(
                            rgb, frame.frame_id, REASON_PAINT, detail
                        )
                    state[frame.top + dy, frame.left + dx] = color_to_id[rgb]
            # 透明像素不动 state：天然表达“沿用画前内容”。
            display_grids.append(state.copy())

            disposal = frame.disposal if frame.disposal != 0 else 1
            if disposal == 2:
                bg_id = const_id(
                    bg,
                    frame.frame_id,
                    REASON_RESTORE_BACKGROUND,
                    f"帧{frame.frame_id}处置2：子矩形恢复片条逻辑背景{bg}",
                )
                state[
                    frame.top : frame.top + frame.height,
                    frame.left : frame.left + frame.width,
                ] = bg_id
            elif disposal == 3:
                state = before  # type: ignore[assignment]

        trace = SymbolicTrace(
            frame_ids=self.frame_ids,
            display_grids=display_grids,
            exit_grid=state.copy(),
            terms=terms,
            starts=starts,
            ends=ends,
        )
        self._trace = trace
        self._trace_fingerprint = fp
        return trace

    # ------------------------------------------------------------------ #
    # 来源追溯：指定时刻显示 / 离场底画，逐像素回答来源与可定位原因。
    # ------------------------------------------------------------------ #
    def provenance_display(self, time_point) -> tuple[np.ndarray, list[list[Term]], int]:
        """返回 (frame_id, Term 二维列表, 帧序号)。"""
        index = self.select_index(time_point)
        if index is None:
            raise FilmstripValidationError("时间点落在片条结束之后，无显示帧")
        return self._provenance_from_grid(self.symbolic_trace().display_grids[index], index)

    def provenance_exit(self) -> list[list[Term]]:
        return self._provenance_from_grid(self.symbolic_trace().exit_grid)[1]

    def _provenance_from_grid(
        self, grid: np.ndarray, index: int | None = None
    ) -> tuple[np.ndarray, list[list[Term]], int]:
        trace = self.symbolic_trace()
        out: list[list[Term]] = []
        for y in range(self.height):
            row: list[Term] = []
            for x in range(self.width):
                row.append(trace.term_at(int(grid[y, x])))
            out.append(row)
        frame_id = self._frames[index].frame_id if index is not None else -1
        return grid, out, frame_id

    def explain_pixel(
        self,
        coord: tuple[int, int],
        *,
        time_point=None,
        at_exit: bool = False,
    ) -> dict:
        """单像素事件链复盘，给出最终来源与可定位原因。

        time_point 给定时解释该时刻显示；at_exit=True 解释全部处置后的离场底画。
        """
        x, y = coord
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise FilmstripValidationError(f"坐标 {coord} 越界")
        if time_point is None and not at_exit:
            raise FilmstripValidationError("必须指定 time_point 或 at_exit=True")
        target_index = None
        if time_point is not None:
            target_index = self.select_index(time_point)
            if target_index is None:
                raise FilmstripValidationError("时间点落在片条结束之后")

        # 标量符号模拟，只跟踪这一个像素，记录每帧事件。
        current = ENTRY_TERM
        events: list[dict] = []
        answer: Term = ENTRY_TERM
        for i, frame in enumerate(self._frames):
            in_rect = (
                frame.left <= x < frame.left + frame.width
                and frame.top <= y < frame.top + frame.height
            )
            before = current
            if in_rect:
                dx, dy = x - frame.left, y - frame.top
                alpha = int(frame.rgba[dy, dx, 3])
                if alpha > 0:
                    rgb = tuple(int(c) for c in frame.rgba[dy, dx, :3])
                    palette_note = "局部色表" if frame.palette_is_local else "帧正文"
                    current = Term(
                        kind="constant",
                        rgb=rgb,
                        frame_id=frame.frame_id,
                        reason=REASON_PAINT,
                        detail=f"帧{frame.frame_id}不透明覆盖，{palette_note}确定颜色 {rgb}",
                    )
                    events.append(
                        {"frame_id": frame.frame_id, "event": REASON_PAINT, "rgb": list(rgb)}
                    )
                else:
                    events.append(
                        {
                            "frame_id": frame.frame_id,
                            "event": REASON_TRANSPARENT,
                            "detail": "透明像素不覆盖，沿用画前内容",
                        }
                    )
            if target_index == i:
                answer = current
            disposal = frame.disposal if frame.disposal != 0 else 1
            if disposal == 2 and in_rect:
                bg = tuple(int(c) for c in self.background_rgb)
                current = Term(
                    kind="constant",
                    rgb=bg,
                    frame_id=frame.frame_id,
                    reason=REASON_RESTORE_BACKGROUND,
                    detail=f"帧{frame.frame_id}处置2：恢复片条逻辑背景 {bg}（非调用方入场色）",
                )
                events.append(
                    {
                        "frame_id": frame.frame_id,
                        "event": REASON_RESTORE_BACKGROUND,
                        "rgb": list(bg),
                    }
                )
            elif disposal == 3:
                current = before
                events.append(
                    {
                        "frame_id": frame.frame_id,
                        "event": REASON_RESTORE_BEFORE,
                        "detail": "处置3：恢复该帧绘制前内容",
                    }
                )
        if at_exit:
            answer = current
        return {
            "coord": [x, y],
            "source": answer.to_dict(),
            "events": events,
        }

    # ------------------------------------------------------------------ #
    # 来源项在具体入场底画下求值
    # ------------------------------------------------------------------ #
    def evaluate_term(
        self,
        term: Term,
        entry_rgba: np.ndarray,
        coord: tuple[int, int],
    ) -> tuple[int, int, int]:
        entry = self._coerce_entry(entry_rgba)
        x, y = coord
        if term.is_entry:
            return tuple(int(c) for c in entry[y, x, :3])  # type: ignore[return-value]
        assert term.rgb is not None
        return term.rgb

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        if self.product_dir is None:
            raise FilmstripValidationError("纯内存片条未绑定 product_dir，无法保存")
        root = Path(self.product_dir)
        (root / "bodies").mkdir(parents=True, exist_ok=True)
        for f in self._frames:
            Image.fromarray(f.rgba, mode="RGBA").save(
                root / "bodies" / f"fid_{f.frame_id}.png", format="PNG"
            )
        payload = {
            "format": STRIP_FORMAT,
            "version": STRIP_VERSION,
            "strip_id": self.strip_id,
            "strip_version": self.strip_version,
            "source_catalog": self.source_catalog,
            "width": self.width,
            "height": self.height,
            "background_rgb": list(self.background_rgb),
            "fingerprint": self.fingerprint(),
            "frames": [
                {
                    "frame_id": f.frame_id,
                    "left": f.left,
                    "top": f.top,
                    "width": f.width,
                    "height": f.height,
                    "delay_centis": f.delay_centis,
                    "disposal": f.disposal,
                    "palette": [list(c) for c in f.palette] if f.palette is not None else None,
                    "palette_is_local": f.palette_is_local,
                    "transparent_index": f.transparent_index,
                    "body": f"bodies/fid_{f.frame_id}.png",
                }
                for f in self._frames
            ],
        }
        with open(root / "strip.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, product_dir: str | os.PathLike[str]) -> "Filmstrip":
        root = Path(product_dir)
        with open(root / "strip.json", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != STRIP_FORMAT or payload.get("version") != STRIP_VERSION:
            raise FilmstripValidationError("片条产品格式/版本不匹配")
        frames: list[RawFrame] = []
        for m in payload["frames"]:
            with Image.open(root / m["body"]) as im:
                rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
            palette = (
                tuple(tuple(c) for c in m["palette"]) if m["palette"] is not None else None
            )
            frames.append(
                RawFrame(
                    frame_id=m["frame_id"],
                    left=m["left"],
                    top=m["top"],
                    width=m["width"],
                    height=m["height"],
                    rgba=rgba,
                    delay_centis=m["delay_centis"],
                    disposal=m["disposal"],
                    palette=palette,
                    palette_is_local=m["palette_is_local"],
                    transparent_index=m["transparent_index"],
                )
            )
        strip = cls(
            payload["strip_id"],
            payload["width"],
            payload["height"],
            tuple(payload["background_rgb"]),
            frames,
            strip_version=payload["strip_version"],
            product_dir=root,
            source_catalog=payload.get("source_catalog"),
        )
        if strip.fingerprint() != payload["fingerprint"]:
            raise FilmstripValidationError("片条正文与指纹不符，产品已被改动")
        return strip


# --------------------------------------------------------------------------- #
# 片条样本生成器：从任意合法入场底画（含逻辑背景入场）产出真实 PNG 样本。
# 原整片从逻辑背景开始的 produce_dataset 仍在 compositor 中，不受影响。
# --------------------------------------------------------------------------- #

def produce_strip_samples(
    strip: Filmstrip,
    output_dir: str | os.PathLike[str],
    entry_rgba: np.ndarray,
    time_points,
) -> dict:
    """消费片条产品，按入场底画流式产出 PNG/JSONL，并附逐像素来源计数。"""
    out = Path(output_dir)
    png_dir = out / "frames"
    png_dir.mkdir(parents=True, exist_ok=True)

    requests = [(f"s{i:06d}", normalize_time(tp)) for i, tp in enumerate(time_points)]
    trace = strip.symbolic_trace()
    digest_to_file: dict[str, str] = {}
    records: list[dict] = []

    for sample_id, t in requests:
        index = strip.select_index(t)
        if index is None:
            records.append(
                {
                    "sample_id": sample_id,
                    "time_seconds": str(float(t)),
                    "time_fraction": str(t),
                    "ended": True,
                    "frame_id": None,
                    "png": None,
                    "rgba_digest": None,
                    "entry_derived_pixels": None,
                    "constant_pixels": None,
                }
            )
            continue
        display = strip.display_at_time(entry_rgba, t)
        assert display is not None
        digest = hashlib.sha256(
            np.ascontiguousarray(display.canvas).tobytes()
        ).hexdigest()
        if digest not in digest_to_file:
            fname = f"{digest[:16]}.png"
            Image.fromarray(display.canvas, mode="RGBA").save(png_dir / fname, format="PNG")
            digest_to_file[digest] = f"frames/{fname}"
        grid = trace.display_grids[index]
        entry_pixels = int((grid == ENTRY_TERM_ID).sum())
        constant_pixels = int((grid != ENTRY_TERM_ID).sum())
        records.append(
            {
                "sample_id": sample_id,
                "time_seconds": str(float(t)),
                "time_fraction": str(t),
                "ended": False,
                "frame_id": display.frame_id,
                "png": digest_to_file[digest],
                "rgba_digest": digest,
                "entry_derived_pixels": entry_pixels,
                "constant_pixels": constant_pixels,
            }
        )

    with open(out / "samples.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "format": "animgif-strip-samples",
        "version": 1,
        "strip_id": strip.strip_id,
        "strip_version": strip.strip_version,
        "strip_fingerprint": strip.fingerprint(),
        "canvas_width": strip.width,
        "canvas_height": strip.height,
        "background_rgb": list(strip.background_rgb),
        "sample_count": len(requests),
        "unique_png_count": len(digest_to_file),
        "loop": False,
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest
