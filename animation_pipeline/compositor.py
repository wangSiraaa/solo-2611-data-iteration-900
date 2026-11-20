"""真实显示帧合成与采样数据集生产。

合成规则（合同口径，不信任任何图像库的默认展示行为）：
* 初始画布为逻辑背景色的完全不透明 RGBA；
* 处置 0 按处置 1 处理：局部 alpha>0 的像素覆盖当前画布，
  透明像素不覆盖旧像素，局部调色板只影响本帧子矩形；
* 先保存“本帧绘制完成的完整显示画布”，再按处置方式准备下一帧：
  - 2：把该帧子矩形恢复为逻辑背景色（不透明）；
  - 3：恢复到绘制本帧之前保存的画布；
* 延时单位百分之一秒，0 延时按 1 处理；
* 半开时间区间 [start, end) 选帧，恰在终点选下一帧；
  只播放一遍，末尾之后返回结束状态，不循环。

检查点只缓存“受处置依赖影响”的关键画布（绘制前快照），
位置不写死：按字节预算 LRU，可由调用方权衡内存与修订成本。
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from .catalog import FrameCatalog, RawFrame

CENTIS = Fraction(1, 100)
DATASET_FORMAT = "animgif-dataset"
DATASET_VERSION = 1


def effective_delay_centis(delay_centis: int) -> int:
    """0 延时按 1 处理。"""
    return delay_centis if delay_centis > 0 else 1


# --------------------------------------------------------------------------- #
# 合成引擎
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DisplayFrame:
    frame_id: int
    start: Fraction       # 本帧显示起点（秒，有理数）
    end: Fraction         # 本帧显示终点（不含）
    canvas: np.ndarray    # 完整逻辑画布 RGBA
    overwritten_pixels: int  # 本帧 alpha>0 覆盖像素数
    transparent_pixels: int  # 本帧透明（未覆盖）像素数


class CheckpointCache:
    """画布检查点的 LRU 缓存；超出预算时淘汰，可选择溢出到临时目录。"""

    def __init__(self, byte_budget: int = 8 * 1024 * 1024, spill_dir: str | os.PathLike[str] | None = None) -> None:
        self.byte_budget = int(byte_budget)
        self.spill_dir = Path(spill_dir) if spill_dir is not None else None
        if self.spill_dir is not None:
            self.spill_dir.mkdir(parents=True, exist_ok=True)
        self._ram: OrderedDict[int, np.ndarray] = OrderedDict()
        self._used = 0
        self._spill_seq = 0

    def get(self, frame_index: int) -> np.ndarray | None:
        if frame_index in self._ram:
            self._ram.move_to_end(frame_index)
            return self._ram[frame_index]
        if self.spill_dir is not None:
            p = self.spill_dir / f"cp_{frame_index}.npy"
            if p.exists():
                arr = np.load(p)
                self.put(frame_index, arr)
                return arr
        return None

    def put(self, frame_index: int, canvas: np.ndarray) -> None:
        nbytes = canvas.nbytes
        if self.spill_dir is not None and nbytes > self.byte_budget:
            # 单帧就超预算：直接落盘，不占 RAM。
            np.save(self.spill_dir / f"cp_{frame_index}.npy", canvas)
            return
        while self._ram and self._used + nbytes > self.byte_budget:
            old_idx, old = self._ram.popitem(last=False)
            self._used -= old.nbytes
            if self.spill_dir is not None:
                np.save(self.spill_dir / f"cp_{old_idx}.npy", old)
        self._ram[frame_index] = canvas.copy()
        self._used += nbytes

    def invalidate_from(self, frame_index: int) -> None:
        """编辑早期帧后，序号 >= frame_index 的检查点全部失效。"""
        for idx in [i for i in self._ram if i >= frame_index]:
            self._used -= self._ram.pop(idx).nbytes
        if self.spill_dir is not None:
            for p in self.spill_dir.glob("cp_*.npy"):
                idx = int(p.stem.split("_")[1])
                if idx >= frame_index:
                    p.unlink()

    def clear(self) -> None:
        self._ram.clear()
        self._used = 0
        if self.spill_dir is not None:
            for p in self.spill_dir.glob("cp_*.npy"):
                p.unlink()


class FramePlayer:
    """按目录原始帧 API 顺序重放显示画布，支持检查点与按时间选择。

    检查点位置不写死：replay 每经过一帧都尝试缓存“下一帧的绘制前画布”，
    由 CheckpointCache 的预算/LRU 自行决定实际保留哪些。
    """

    def __init__(
        self,
        catalog: FrameCatalog,
        checkpoints: CheckpointCache | None = None,
    ) -> None:
        self.catalog = catalog
        self.checkpoints = checkpoints or CheckpointCache(byte_budget=0)
        self._frames_meta = catalog.list_metadata()
        self._starts: list[Fraction] | None = None
        self._ends: list[Fraction] | None = None
        self._revision_at_build = catalog.revision

    # -- 时间轴 ---------------------------------------------------------- #
    def _ensure_timeline(self) -> None:
        if self._starts is not None and self._revision_at_build == self.catalog.revision:
            return
        starts: list[Fraction] = []
        ends: list[Fraction] = []
        t = Fraction(0)
        for meta in self._frames_meta:
            starts.append(t)
            t += effective_delay_centis(meta["delay_centis"]) * CENTIS
            ends.append(t)
        self._starts, self._ends = starts, ends
        self._revision_at_build = self.catalog.revision

    @property
    def end_time(self) -> Fraction:
        self._ensure_timeline()
        return self._ends[-1] if self._ends else Fraction(0)

    def select_index(self, time_point: Fraction) -> int | None:
        """半开区间选帧：终点落在下一帧；末尾之后返回 None（结束）。"""
        self._ensure_timeline()
        if not self._starts or time_point >= self._ends[-1]:
            return None
        # bisect_right(starts, t) - 1：t 恰为某起点时选该帧。
        idx = bisect.bisect_right(self._starts, time_point) - 1
        return max(idx, 0)

    # -- 重放 ------------------------------------------------------------ #
    def _initial_canvas(self) -> np.ndarray:
        canvas = np.zeros((self.catalog.height, self.catalog.width, 4), dtype=np.uint8)
        canvas[:, :, :3] = np.asarray(self.catalog.background_rgb, dtype=np.uint8)
        canvas[:, :, 3] = 255
        return canvas

    def _draw(self, canvas: np.ndarray, frame: RawFrame) -> tuple[int, int]:
        """把局部正文覆盖到画布，返回 (覆盖像素数, 透明像素数)。"""
        region = canvas[frame.top : frame.top + frame.height, frame.left : frame.left + frame.width]
        alpha = frame.rgba[:, :, 3]
        mask = alpha > 0
        region[mask] = frame.rgba[mask]
        overwritten = int(mask.sum())
        transparent = int((~mask).sum())
        return overwritten, transparent

    def _restore_background(self, canvas: np.ndarray, frame: RawFrame) -> None:
        region = canvas[frame.top : frame.top + frame.height, frame.left : frame.left + frame.width]
        region[:, :, :3] = np.asarray(self.catalog.background_rgb, dtype=np.uint8)
        region[:, :, 3] = 255

    def replay(self, stop_index: int | None = None):
        """从头顺序重放，逐帧产出 DisplayFrame（这是唯一的事实来源）。

        每帧绘制前的画布会尝试写入检查点缓存（是否实际保留由预算决定），
        使 display_at 可以从最近检查点恢复而不必总是从头重放。
        """
        self._ensure_timeline()
        limit = len(self._frames_meta) if stop_index is None else stop_index
        canvas = self._initial_canvas()
        for i, frame in enumerate(self.catalog.iter_raw_frames()):
            if i >= limit:
                break
            self.checkpoints.put(i, canvas)
            before = canvas.copy() if frame.disposal == 3 else None
            overwritten, transparent = self._draw(canvas, frame)
            yield DisplayFrame(
                frame_id=frame.frame_id,
                start=self._starts[i],
                end=self._ends[i],
                canvas=canvas.copy(),
                overwritten_pixels=overwritten,
                transparent_pixels=transparent,
            )
            # 保存显示画布之后，再按处置方式准备下一帧画布。
            disposal = frame.disposal if frame.disposal != 0 else 1
            if disposal == 2:
                self._restore_background(canvas, frame)
            elif disposal == 3:
                canvas = before  # type: ignore[assignment]
            # 处置 1/0：保留当前画布。

    def display_at(self, index: int) -> DisplayFrame:
        """按序号取显示帧；从最近可用检查点恢复，缺失部分顺序重放。"""
        self._ensure_timeline()
        if not 0 <= index < len(self._frames_meta):
            raise IndexError("帧序号越界")
        cp_index = index
        while cp_index > 0 and self.checkpoints.get(cp_index) is None:
            cp_index -= 1
        canvas = (
            self._initial_canvas()
            if cp_index == 0
            else self.checkpoints.get(cp_index).copy()  # type: ignore[union-attr]
        )
        result: DisplayFrame | None = None
        raw_frames = list(self.catalog.iter_raw_frames())
        for i in range(cp_index, index + 1):
            frame = raw_frames[i]
            self.checkpoints.put(i, canvas)
            before = canvas.copy() if frame.disposal == 3 else None
            overwritten, transparent = self._draw(canvas, frame)
            if i == index:
                result = DisplayFrame(
                    frame_id=frame.frame_id,
                    start=self._starts[i],
                    end=self._ends[i],
                    canvas=canvas.copy(),
                    overwritten_pixels=overwritten,
                    transparent_pixels=transparent,
                )
            disposal = frame.disposal if frame.disposal != 0 else 1
            if disposal == 2:
                self._restore_background(canvas, frame)
            elif disposal == 3:
                canvas = before  # type: ignore[assignment]
        assert result is not None
        return result


# --------------------------------------------------------------------------- #
# 独立参考实现：纯 Python 二维数组重放（测试中逐像素核对 numpy 实现）
# --------------------------------------------------------------------------- #

def reference_replay(catalog: FrameCatalog) -> list[dict]:
    """不使用 numpy 的逐像素重放，返回每帧画布（嵌套 list）与统计。"""
    w, h = catalog.width, catalog.height
    bg = list(catalog.background_rgb)
    canvas = [[bg + [255] for _ in range(w)] for _ in range(h)]
    out: list[dict] = []
    t = 0
    for frame in catalog.iter_raw_frames():
        before = [row[:] for row in canvas] if frame.disposal == 3 else None
        body = frame.rgba.tolist()
        overwritten = 0
        transparent = 0
        for y in range(frame.height):
            for x in range(frame.width):
                r, g, b, a = body[y][x]
                if a > 0:
                    canvas[frame.top + y][frame.left + x] = [r, g, b, a]
                    overwritten += 1
                else:
                    transparent += 1
        delay = effective_delay_centis(frame.delay_centis)
        out.append(
            {
                "frame_id": frame.frame_id,
                "start": t,
                "end": t + delay,
                "canvas": [row[:] for row in canvas],
                "overwritten_pixels": overwritten,
                "transparent_pixels": transparent,
            }
        )
        t += delay
        disposal = frame.disposal if frame.disposal != 0 else 1
        if disposal == 2:
            for y in range(frame.top, frame.top + frame.height):
                for x in range(frame.left, frame.left + frame.width):
                    canvas[y][x] = bg + [255]
        elif disposal == 3:
            canvas = before  # type: ignore[assignment]
    return out


# --------------------------------------------------------------------------- #
# 数据集生产
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SampleRequest:
    sample_id: str
    time_point: Fraction


def normalize_time(value) -> Fraction:
    """接受 int/Fraction/浮点字符串/分数字符串，统一为非负 Fraction 秒。"""
    if isinstance(value, Fraction):
        t = value
    elif isinstance(value, int):
        t = Fraction(value)
    elif isinstance(value, float):
        # 浮点按最短十进制表示转分数，避免隐式二进制误差。
        t = Fraction(repr(value))
    elif isinstance(value, str):
        t = Fraction(value)
    else:
        raise TypeError(f"不支持的时间点类型 {type(value)!r}")
    if t < 0:
        raise ValueError("时间点不能为负")
    return t


def rgba_digest(canvas: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(canvas).tobytes()).hexdigest()


def produce_dataset(
    catalog: FrameCatalog,
    output_dir: str | os.PathLike[str],
    time_points,
    *,
    checkpoint_budget_bytes: int = 8 * 1024 * 1024,
    spill_dir: str | os.PathLike[str] | None = None,
) -> dict:
    """消费目录原始帧 API，流式产出真实 PNG 与 JSONL。

    同一显示画布（RGBA 摘要相同）的多个采样共享一个 PNG，
    但每个采样在 JSONL 中保留独立样本身份。
    """
    if catalog.frame_count() == 0:
        raise ValueError("目录为空，无法生产数据集")
    out = Path(output_dir)
    png_dir = out / "frames"
    png_dir.mkdir(parents=True, exist_ok=True)

    requests = [
        SampleRequest(
            sample_id=f"s{i:06d}",
            time_point=normalize_time(tp),
        )
        for i, tp in enumerate(time_points)
    ]
    order = sorted(range(len(requests)), key=lambda k: requests[k].time_point)

    player = FramePlayer(
        catalog,
        CheckpointCache(byte_budget=checkpoint_budget_bytes, spill_dir=spill_dir),
    )
    total_pixels = catalog.width * catalog.height

    # 单次顺序扫描：时间有序，重放一遍即可覆盖全部采样。
    records: list[dict] = [None] * len(requests)  # type: ignore[list-item]
    digest_to_file: dict[str, str] = {}
    display_iter = player.replay()
    current: DisplayFrame | None = next(display_iter, None)

    for req_idx in order:
        req = requests[req_idx]
        # 半开区间：到达当前帧终点就推进；恰在终点因此落入下一帧。
        while current is not None and req.time_point >= current.end:
            current = next(display_iter, None)

        if current is None:
            records[req_idx] = {
                "sample_id": req.sample_id,
                "time_seconds": _fraction_str(req.time_point),
                "time_fraction": str(req.time_point),
                "ended": True,
                "frame_id": None,
                "png": None,
                "rgba_digest": None,
                "overwritten_pixels": None,
                "inherited_pixels": None,
                "transparent_pixels": None,
                "canvas_width": catalog.width,
                "canvas_height": catalog.height,
            }
            continue

        digest = rgba_digest(current.canvas)
        if digest not in digest_to_file:
            fname = f"{digest[:16]}.png"
            Image.fromarray(current.canvas, mode="RGBA").save(png_dir / fname, format="PNG")
            digest_to_file[digest] = f"frames/{fname}"
        records[req_idx] = {
            "sample_id": req.sample_id,
            "time_seconds": _fraction_str(req.time_point),
            "time_fraction": str(req.time_point),
            "ended": False,
            "frame_id": current.frame_id,
            "png": digest_to_file[digest],
            "rgba_digest": digest,
            "overwritten_pixels": current.overwritten_pixels,
            "inherited_pixels": total_pixels - current.overwritten_pixels,
            "transparent_pixels": current.transparent_pixels,
            "canvas_width": catalog.width,
            "canvas_height": catalog.height,
        }

    jsonl_path = out / "samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "format": DATASET_FORMAT,
        "version": DATASET_VERSION,
        "catalog_revision": catalog.revision,
        "canvas_width": catalog.width,
        "canvas_height": catalog.height,
        "background_rgb": list(catalog.background_rgb),
        "sample_count": len(requests),
        "frame_count": catalog.frame_count(),
        "end_time_seconds": _fraction_str(player.end_time),
        "unique_png_count": len(digest_to_file),
        "loop": False,
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def _fraction_str(value: Fraction) -> str:
    """有理数转十进制字符串（有限/循环都保留精确分数另存，本字段给人读）。"""
    return str(float(value))
