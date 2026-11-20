"""可编辑的动画帧目录服务。

目录是脱离数据集独立存在的“产品”：保存逻辑画布大小、逻辑背景 RGB、
每帧子矩形、RGBA 正文、百分之一秒整数延时与处置方式，并保留 GIF 来源的
索引图/调色板作为调色板修订的依据。

关键约束：
* 每帧有稳定整数身份 frame_id，删除后该 id 永不复用；
* 帧正文以 PNG 落盘，内存只保留字节预算内的 LRU 缓存（默认 12 MiB）；
* 修改矩形验证画布边界与正文形状，修改调色板验证索引范围并重渲染正文；
* 提供 *不合成画布* 的原始帧 API（iter_raw_frames / get_frame），
  数据集生产器只能消费它，不允许直接打开原 GIF 取库生成的最终帧。
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .gif_io import GifFile, parse_gif

CATALOG_FORMAT = "animgif-catalog"
CATALOG_VERSION = 1
DEFAULT_BODY_CACHE_BYTES = 12 * 1024 * 1024
VALID_DISPOSALS = (0, 1, 2, 3)


class CatalogValidationError(ValueError):
    """矩形越界、索引越界或参数非法时抛出。"""


@dataclass(frozen=True)
class RawFrame:
    """一帧的原始素材（局部 RGBA 正文 + 元数据），不含任何合成结果。"""

    frame_id: int
    left: int
    top: int
    width: int
    height: int
    rgba: np.ndarray  # shape=(height, width, 4) dtype=uint8
    delay_centis: int
    disposal: int
    palette: tuple[tuple[int, int, int], ...] | None
    palette_is_local: bool
    transparent_index: int | None

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.width, self.height


def _validate_rect(rect: tuple[int, int, int, int], canvas_w: int, canvas_h: int) -> tuple[int, int, int, int]:
    if len(rect) != 4 or any(not isinstance(v, int) for v in rect):
        raise CatalogValidationError("rect 必须是 4 个整数 (left, top, width, height)")
    left, top, width, height = rect
    if width <= 0 or height <= 0:
        raise CatalogValidationError("矩形宽高必须为正整数")
    if left < 0 or top < 0 or left + width > canvas_w or top + height > canvas_h:
        raise CatalogValidationError(
            f"矩形 {rect} 越界（画布 {canvas_w}x{canvas_h}）"
        )
    return left, top, width, height


def _validate_rgba(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(rgba)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 4:
        raise CatalogValidationError("rgba 必须是 dtype=uint8 的 HxWx4 数组")
    if arr.shape != (height, width, 4):
        raise CatalogValidationError(
            f"正文形状 {arr.shape} 与矩形 {width}x{height} 不符"
        )
    return arr


class FrameCatalog:
    """帧目录产品。可纯内存使用，也可绑定目录持久化。"""

    def __init__(
        self,
        width: int,
        height: int,
        background_rgb: tuple[int, int, int],
        product_dir: str | os.PathLike[str] | None = None,
        body_cache_bytes: int = DEFAULT_BODY_CACHE_BYTES,
    ) -> None:
        if width <= 0 or height <= 0:
            raise CatalogValidationError("画布宽高必须为正整数")
        if len(background_rgb) != 3 or any(not 0 <= c <= 255 for c in background_rgb):
            raise CatalogValidationError("背景 RGB 必须是 3 个 0..255 整数")
        self.width = int(width)
        self.height = int(height)
        self.background_rgb = tuple(int(c) for c in background_rgb)
        self.product_dir = Path(product_dir) if product_dir is not None else None
        self._body_cache_bytes = int(body_cache_bytes)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_used = 0
        # 元数据顺序即帧顺序；frame_id 是稳定身份。
        self._meta: list[dict] = []
        self._by_id: dict[int, dict] = {}
        self._next_frame_id = 0
        self._deleted_ids: set[int] = set()
        self._revision = 1
        # 纯内存目录没有磁盘兜底，正文必须全部驻留；
        # 绑定目录的产品按字节预算做 LRU，淘汰后可从 PNG 重新载入。
        self._memory_bodies: dict[int, np.ndarray] = {}
        if self.product_dir is not None:
            (self.product_dir / "bodies").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 从结构解析后的 GIF 导入（不读取任何合成结果）
    # ------------------------------------------------------------------ #
    @classmethod
    def from_gif(
        cls,
        gif_path: str | os.PathLike[str],
        product_dir: str | os.PathLike[str] | None,
        body_cache_bytes: int = DEFAULT_BODY_CACHE_BYTES,
    ) -> "FrameCatalog":
        gif: GifFile = parse_gif(os.fspath(gif_path))
        if gif.global_palette:
            bg_rgb = gif.global_palette[gif.background_index]
        else:
            bg_rgb = (0, 0, 0)
        catalog = cls(
            gif.width,
            gif.height,
            bg_rgb,
            product_dir,
            body_cache_bytes=body_cache_bytes,
        )
        for img in gif.images:
            palette = tuple(img.palette)
            max_index = max(img.indices) if img.indices else 0
            if max_index >= len(palette):
                raise CatalogValidationError(
                    f"帧 {img.index} 出现调色板外索引 {max_index}"
                )
            if img.transparent_index is not None and not 0 <= img.transparent_index < len(palette):
                raise CatalogValidationError(
                    f"帧 {img.index} 透明索引 {img.transparent_index} 越界"
                )
            rgba = np.empty((img.height, img.width, 4), dtype=np.uint8)
            flat = rgba.reshape(-1, 4)
            pal = np.asarray(palette, dtype=np.uint8)
            idx_arr = np.frombuffer(img.indices, dtype=np.uint8)
            flat[:, :3] = pal[idx_arr]
            flat[:, 3] = 255
            if img.transparent_index is not None:
                flat[idx_arr == img.transparent_index, 3] = 0
            catalog.add_frame(
                rect=(img.left, img.top, img.width, img.height),
                rgba=rgba,
                delay_centis=img.delay_centis,
                disposal=img.disposal,
                palette=palette,
                palette_is_local=img.palette_is_local,
                transparent_index=img.transparent_index,
                indices=bytes(img.indices),
            )
        catalog._revision = 1
        if catalog.product_dir is not None:
            catalog.save()
        return catalog

    # ------------------------------------------------------------------ #
    # 增删改
    # ------------------------------------------------------------------ #
    def add_frame(
        self,
        rect: tuple[int, int, int, int],
        rgba: np.ndarray,
        delay_centis: int,
        disposal: int,
        *,
        palette: tuple[tuple[int, int, int], ...] | None = None,
        palette_is_local: bool = False,
        transparent_index: int | None = None,
        indices: bytes | None = None,
    ) -> int:
        if not isinstance(delay_centis, int) or delay_centis < 0:
            raise CatalogValidationError("延时必须是非负整数（百分之一秒）")
        if disposal not in VALID_DISPOSALS:
            raise CatalogValidationError(f"处置方式必须是 {VALID_DISPOSALS} 之一")
        left, top, width, height = _validate_rect(rect, self.width, self.height)
        body = _validate_rgba(rgba, width, height)
        if palette is not None:
            self._check_palette(palette, transparent_index, indices, width * height)
        frame_id = self._next_frame_id
        self._next_frame_id += 1
        meta = {
            "frame_id": frame_id,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "delay_centis": int(delay_centis),
            "disposal": int(disposal),
            "palette": [tuple(c) for c in palette] if palette is not None else None,
            "palette_is_local": bool(palette_is_local),
            "transparent_index": transparent_index,
            "has_indices": indices is not None,
        }
        self._meta.append(meta)
        self._by_id[frame_id] = meta
        if indices is not None:
            self._write_indices(frame_id, indices)
        self._write_body(frame_id, body)
        self._bump_revision()
        return frame_id

    def delete_frame(self, frame_id: int) -> None:
        meta = self._require_meta(frame_id)
        self._meta.remove(meta)
        del self._by_id[frame_id]
        self._deleted_ids.add(frame_id)
        self._cache.pop(frame_id, None)
        self._memory_bodies.pop(frame_id, None)
        if self.product_dir is not None:
            for suffix in (".png", ".idx"):
                p = self.product_dir / "bodies" / f"fid_{frame_id}{suffix}"
                if p.exists():
                    p.unlink()
        self._bump_revision()

    def replace_body(self, frame_id: int, rgba: np.ndarray) -> None:
        meta = self._require_meta(frame_id)
        body = _validate_rgba(rgba, meta["width"], meta["height"])
        self._write_body(frame_id, body)
        self._bump_revision()

    def replace_rect(self, frame_id: int, rect: tuple[int, int, int, int]) -> None:
        """只改矩形：新尺寸必须与现有正文一致，否则请先 replace_body。"""
        meta = self._require_meta(frame_id)
        left, top, width, height = _validate_rect(rect, self.width, self.height)
        if (width, height) != (meta["width"], meta["height"]):
            raise CatalogValidationError(
                "矩形尺寸变化时必须同时提供匹配的新 RGBA 正文"
            )
        meta.update(left=left, top=top, width=width, height=height)
        self._bump_revision()

    def replace_rect_and_body(
        self, frame_id: int, rect: tuple[int, int, int, int], rgba: np.ndarray
    ) -> None:
        meta = self._require_meta(frame_id)
        left, top, width, height = _validate_rect(rect, self.width, self.height)
        body = _validate_rgba(rgba, width, height)
        meta.update(left=left, top=top, width=width, height=height)
        self._write_body(frame_id, body)
        self._bump_revision()

    def update_delay(self, frame_id: int, delay_centis: int) -> None:
        meta = self._require_meta(frame_id)
        if not isinstance(delay_centis, int) or delay_centis < 0:
            raise CatalogValidationError("延时必须是非负整数（百分之一秒）")
        meta["delay_centis"] = delay_centis
        self._bump_revision()

    def update_disposal(self, frame_id: int, disposal: int) -> None:
        meta = self._require_meta(frame_id)
        if disposal not in VALID_DISPOSALS:
            raise CatalogValidationError(f"处置方式必须是 {VALID_DISPOSALS} 之一")
        meta["disposal"] = disposal
        self._bump_revision()

    def update_palette(
        self,
        frame_id: int,
        palette: tuple[tuple[int, int, int], ...],
        transparent_index: int | None = None,
    ) -> None:
        """修订调色板引用：验证全部索引后按新调色板重渲染 RGBA 正文。"""
        meta = self._require_meta(frame_id)
        if not meta["has_indices"]:
            raise CatalogValidationError(
                f"帧 {frame_id} 没有保留索引图，无法按调色板重渲染；"
                "可使用 replace_body 直接替换 RGBA 正文"
            )
        indices = self._read_indices(frame_id)
        self._check_palette(palette, transparent_index, indices, meta["width"] * meta["height"])
        rgba = self._render_rgba(indices, palette, transparent_index, meta["width"], meta["height"])
        meta["palette"] = [tuple(c) for c in palette]
        meta["transparent_index"] = transparent_index
        self._write_body(frame_id, rgba)
        self._bump_revision()

    # ------------------------------------------------------------------ #
    # 读取：原始帧 API（不合成）
    # ------------------------------------------------------------------ #
    @property
    def revision(self) -> int:
        return self._revision

    @property
    def frame_ids(self) -> list[int]:
        return [m["frame_id"] for m in self._meta]

    def frame_count(self) -> int:
        return len(self._meta)

    def list_metadata(self) -> tuple[dict, ...]:
        # 返回副本，避免外部绕过校验改元数据。
        return tuple(dict(m) for m in self._meta)

    def get_frame(self, frame_id: int) -> RawFrame:
        meta = self._require_meta(frame_id)
        return self._to_raw_frame(meta, self._load_body(frame_id))

    def iter_raw_frames(self):
        """按帧顺序产出 RawFrame；数据集生产器必须通过本 API 取素材。"""
        for meta in list(self._meta):
            yield self._to_raw_frame(meta, self._load_body(meta["frame_id"]))

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        if self.product_dir is None:
            raise CatalogValidationError("纯内存目录未绑定 product_dir，无法保存")
        self.product_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CATALOG_FORMAT,
            "version": CATALOG_VERSION,
            "revision": self._revision,
            "width": self.width,
            "height": self.height,
            "background_rgb": list(self.background_rgb),
            "next_frame_id": self._next_frame_id,
            "deleted_frame_ids": sorted(self._deleted_ids),
            "body_cache_bytes_limit": self._body_cache_bytes,
            "frames": [dict(m, body=f"bodies/fid_{m['frame_id']}.png",
                            indices=(f"bodies/fid_{m['frame_id']}.idx" if m["has_indices"] else None))
                       for m in self._meta],
        }
        with open(self.product_dir / "catalog.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def publish_to(self, target_dir: str | os.PathLike[str]) -> "FrameCatalog":
        """把当前目录产品完整发布到新目录（用于修订后发布新产品）。

        复制全部正文 PNG 与索引图后保存 catalog.json，返回绑定新目录的自身。
        """
        import shutil

        target = Path(target_dir)
        (target / "bodies").mkdir(parents=True, exist_ok=True)
        for meta in self._meta:
            fid = meta["frame_id"]
            if self.product_dir is None:
                Image.fromarray(self._memory_bodies[fid], mode="RGBA").save(
                    target / "bodies" / f"fid_{fid}.png", format="PNG"
                )
                if meta["has_indices"]:
                    raise CatalogValidationError("纯内存目录不应携带索引图")
            else:
                shutil.copyfile(
                    self.product_dir / "bodies" / f"fid_{fid}.png",
                    target / "bodies" / f"fid_{fid}.png",
                )
                if meta["has_indices"]:
                    shutil.copyfile(
                        self.product_dir / "bodies" / f"fid_{fid}.idx",
                        target / "bodies" / f"fid_{fid}.idx",
                    )
        self.product_dir = target
        self.save()
        return self

    @classmethod
    def load(cls, product_dir: str | os.PathLike[str]) -> "FrameCatalog":
        root = Path(product_dir)
        with open(root / "catalog.json", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != CATALOG_FORMAT or payload.get("version") != CATALOG_VERSION:
            raise CatalogValidationError("目录产品格式/版本不匹配")
        catalog = cls(
            payload["width"],
            payload["height"],
            tuple(payload["background_rgb"]),
            root,
            body_cache_bytes=payload.get("body_cache_bytes_limit", DEFAULT_BODY_CACHE_BYTES),
        )
        catalog._revision = payload["revision"]
        catalog._next_frame_id = payload["next_frame_id"]
        catalog._deleted_ids = set(payload.get("deleted_frame_ids", ()))
        for m in payload["frames"]:
            meta = {k: m[k] for k in (
                "frame_id", "left", "top", "width", "height", "delay_centis",
                "disposal", "palette", "palette_is_local", "transparent_index",
                "has_indices",
            )}
            catalog._meta.append(meta)
            catalog._by_id[meta["frame_id"]] = meta
        return catalog

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _bump_revision(self) -> None:
        self._revision += 1

    def _require_meta(self, frame_id: int) -> dict:
        if frame_id not in self._by_id:
            raise CatalogValidationError(f"frame_id={frame_id} 不存在或已删除")
        return self._by_id[frame_id]

    @staticmethod
    def _check_palette(
        palette: tuple[tuple[int, int, int], ...],
        transparent_index: int | None,
        indices: bytes | None,
        pixel_count: int,
    ) -> None:
        if not palette or not (1 <= len(palette) <= 256):
            raise CatalogValidationError("调色板必须含 1..256 个条目")
        for color in palette:
            if len(color) != 3 or any(not 0 <= int(c) <= 255 for c in color):
                raise CatalogValidationError("调色板条目必须是 0..255 的 RGB 三元组")
        if transparent_index is not None and not 0 <= transparent_index < len(palette):
            raise CatalogValidationError(
                f"透明索引 {transparent_index} 超出调色板长度 {len(palette)}"
            )
        if indices is not None:
            if len(indices) != pixel_count:
                raise CatalogValidationError("索引图长度与矩形面积不符")
            if indices and max(indices) >= len(palette):
                raise CatalogValidationError("索引图存在调色板外索引")

    @staticmethod
    def _render_rgba(
        indices: bytes,
        palette: tuple[tuple[int, int, int], ...],
        transparent_index: int | None,
        width: int,
        height: int,
    ) -> np.ndarray:
        pal = np.asarray(palette, dtype=np.uint8)
        idx_arr = np.frombuffer(indices, dtype=np.uint8)
        rgba = np.empty((height * width, 4), dtype=np.uint8)
        rgba[:, :3] = pal[idx_arr]
        rgba[:, 3] = 255
        if transparent_index is not None:
            rgba[idx_arr == transparent_index, 3] = 0
        return rgba.reshape(height, width, 4)

    def _body_path(self, frame_id: int) -> Path | None:
        if self.product_dir is None:
            return None
        return self.product_dir / "bodies" / f"fid_{frame_id}.png"

    def _indices_path(self, frame_id: int) -> Path:
        assert self.product_dir is not None
        return self.product_dir / "bodies" / f"fid_{frame_id}.idx"

    def _write_indices(self, frame_id: int, indices: bytes) -> None:
        if self.product_dir is not None:
            with open(self._indices_path(frame_id), "wb") as fh:
                fh.write(indices)

    def _read_indices(self, frame_id: int) -> bytes:
        if self.product_dir is None:
            raise CatalogValidationError("纯内存目录不保留索引图")
        with open(self._indices_path(frame_id), "rb") as fh:
            return fh.read()

    def _write_body(self, frame_id: int, body: np.ndarray) -> None:
        path = self._body_path(frame_id)
        if path is not None:
            Image.fromarray(body, mode="RGBA").save(path, format="PNG")
        else:
            self._memory_bodies[frame_id] = body.copy()
        # 写入后刷新缓存。
        self._cache.pop(frame_id, None)
        self._store_cache(frame_id, body)

    def _load_body(self, frame_id: int) -> np.ndarray:
        if frame_id in self._cache:
            self._cache.move_to_end(frame_id)
            return self._cache[frame_id]
        path = self._body_path(frame_id)
        if path is None:
            body = self._memory_bodies[frame_id].copy()
        elif path.exists():
            with Image.open(path) as im:
                body = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
        else:
            raise CatalogValidationError("目录正文缺失（内部错误）")
        self._store_cache(frame_id, body)
        return body

    def _store_cache(self, frame_id: int, body: np.ndarray) -> None:
        nbytes = body.nbytes
        while self._cache_used + nbytes > self._body_cache_bytes and self._cache:
            _, dropped = self._cache.popitem(last=False)
            self._cache_used -= dropped.nbytes
        self._cache[frame_id] = body
        self._cache_used += nbytes

    def _to_raw_frame(self, meta: dict, rgba: np.ndarray) -> RawFrame:
        return RawFrame(
            frame_id=meta["frame_id"],
            left=meta["left"],
            top=meta["top"],
            width=meta["width"],
            height=meta["height"],
            rgba=rgba,
            delay_centis=meta["delay_centis"],
            disposal=meta["disposal"],
            palette=tuple(tuple(c) for c in meta["palette"]) if meta["palette"] is not None else None,
            palette_is_local=meta["palette_is_local"],
            transparent_index=meta["transparent_index"],
        )
