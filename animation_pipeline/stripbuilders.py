"""片条/展柜演示素材生成器。

只构造目录层已经解析好的 RGBA 原始帧（含透明、局部色表与四种处置），
再通过 Filmstrip.from_catalog 切出固定片条产品。所有素材都小到可手算逐像素核对。

颜色约定（白底/红底差异的标准件）：
    W = (255,255,255) 白；R = (255,0,0) 红。
"""

from __future__ import annotations

import numpy as np

from .catalog import FrameCatalog
from .filmstrip import Filmstrip

W = (255, 255, 255)
R = (255, 0, 0)


def solid_body(width: int, height: int, rgb, *, transparent: bool = False) -> np.ndarray:
    body = np.zeros((height, width, 4), dtype=np.uint8)
    body[:, :, :3] = rgb
    body[:, :, 3] = 0 if transparent else 255
    return body


def _single_frame_catalog(
    width: int,
    height: int,
    background_rgb,
    rgb,
    *,
    transparent: bool,
    disposal: int,
    delay_centis: int = 10,
    palette=None,
    palette_is_local: bool = False,
    transparent_index: int | None = None,
    indices: bytes | None = None,
) -> FrameCatalog:
    catalog = FrameCatalog(width, height, background_rgb)
    catalog.add_frame(
        rect=(0, 0, width, height),
        rgba=solid_body(width, height, rgb, transparent=transparent),
        delay_centis=delay_centis,
        disposal=disposal,
        palette=palette,
        palette_is_local=palette_is_local,
        transparent_index=transparent_index,
        indices=indices,
    )
    return catalog


# --------------------------------------------------------------------------- #
# 标准件 A：白逻辑背景单帧片
#   甲片只显示透明一拍（透明不覆盖，显示即入场色）；
#   乙片画白一拍并保留（处置1），显示恒白、离场恒白。
# --------------------------------------------------------------------------- #

def build_white_pair(size: int = 1):
    bg = W
    cat_a = _single_frame_catalog(size, size, bg, W, transparent=True, disposal=1)
    cat_b = _single_frame_catalog(size, size, bg, W, transparent=False, disposal=1)
    strip_a = Filmstrip.from_catalog(cat_a, cat_a.frame_ids, "white-transparent")
    strip_b = Filmstrip.from_catalog(cat_b, cat_b.frame_ids, "white-paint-keep")
    return strip_a, strip_b


# --------------------------------------------------------------------------- #
# 标准件 B：均显示红的单帧片，区别只在最后处置
#   处置3：画完恢复绘制前内容（离场=入场）；
#   处置1：画完保留（离场=红）。两片显示恒红，只验显示可换，要求离场同色则不能换。
# --------------------------------------------------------------------------- #

def build_red_pair(size: int = 1):
    bg = W
    cat_d3 = _single_frame_catalog(size, size, bg, R, transparent=False, disposal=3)
    cat_d1 = _single_frame_catalog(size, size, bg, R, transparent=False, disposal=1)
    strip_d3 = Filmstrip.from_catalog(cat_d3, cat_d3.frame_ids, "red-paint-restore-before")
    strip_d1 = Filmstrip.from_catalog(cat_d1, cat_d1.frame_ids, "red-paint-keep")
    return strip_d3, strip_d1


# --------------------------------------------------------------------------- #
# 标准件 C：带局部色表的改稿素材。
# 同一索引图 [1]，旧局部表索引1=白，改稿后索引1=红；保留 .idx 以便 update_palette
# 重渲染。签单可在改稿前签发，改稿后重开复读并识别指纹漂移。
# --------------------------------------------------------------------------- #

def build_local_palette_strip(size: int = 1, product_dir=None):
    palette_v1 = (R, W)          # 索引0=红，索引1=白
    catalog = FrameCatalog(size, size, W, product_dir=product_dir)
    catalog.add_frame(
        rect=(0, 0, size, size),
        rgba=solid_body(size, size, W, transparent=False),
        delay_centis=10,
        disposal=1,
        palette=palette_v1,
        palette_is_local=True,
        indices=bytes([1] * (size * size)),
    )
    strip = Filmstrip.from_catalog(catalog, catalog.frame_ids, "local-palette-revision")
    return catalog, strip


def revise_local_palette_to_red(catalog: FrameCatalog, frame_id: int) -> None:
    """一次改稿：局部色表索引 1 由白改红，并按索引图重渲染正文。"""
    palette_v2 = (R, R)
    catalog.update_palette(frame_id, palette_v2, transparent_index=None)


# --------------------------------------------------------------------------- #
# 标准件 D：两区柜演示用片（X 上透明的甲片 / X 上画白的乙片）。
# 片条画布与展柜一致，帧覆盖整画布；柜方区域 X/Y 由 cabinet 演示代码划分。
# --------------------------------------------------------------------------- #

def build_xy_pair(width: int = 2, height: int = 1):
    """两区柜演示片：帧子矩形只覆盖 X 列（(0,0,1,1)），Y 列两片都不触碰。

    甲片在 X 上只显示透明一拍（X 显示=入场色）；乙片在 X 上画白并保留。
    两条片对 Y 都是入场延续，因此替片差异只可能出现在 X。
    """
    bg = W
    cat_a = FrameCatalog(width, height, bg)
    cat_a.add_frame(
        rect=(0, 0, 1, 1),
        rgba=solid_body(1, 1, W, transparent=True),
        delay_centis=10,
        disposal=1,
    )
    cat_b = FrameCatalog(width, height, bg)
    cat_b.add_frame(
        rect=(0, 0, 1, 1),
        rgba=solid_body(1, 1, W, transparent=False),
        delay_centis=10,
        disposal=1,
    )
    strip_a = Filmstrip.from_catalog(cat_a, cat_a.frame_ids, "xy-transparent")
    strip_b = Filmstrip.from_catalog(cat_b, cat_b.frame_ids, "xy-paint-white")
    return strip_a, strip_b
