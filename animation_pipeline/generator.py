"""自行生成真实 GIF 测试输入（只走本仓库 gif_io 写出）。

build_demo_gif 覆盖合同要求的全部特性：
透明叠层、局部调色板同索引不同 RGB、连续处置 3、处置 2 的非整画布、
0 延时以及可用于演示“编辑早期帧后重算”的素材。
build_scale_gif 生成 512x512 / 1000 帧规模输入。
"""

from __future__ import annotations

import random

from .gif_io import frame_spec, write_gif

# 8 色全局调色板（索引 7 留作透明色）。
GLOBAL_PALETTE = [
    (0, 0, 0),        # 0 黑
    (255, 255, 255),  # 1 白
    (255, 0, 0),      # 2 红
    (0, 255, 0),      # 3 绿
    (0, 0, 255),      # 4 蓝
    (255, 255, 0),    # 5 黄
    (255, 128, 0),    # 6 橙
    (128, 128, 128),  # 7 灰（在各帧按需声明为透明）
]
TRANSPARENT = 7


def _rect_indices(width: int, height: int, fill: int) -> list[int]:
    return [fill] * (width * height)


def build_demo_gif(path: str, size: int = 8) -> dict:
    """8x8 小画布，手工可逐像素核对。返回各帧设计说明。"""
    bg_index = 0
    frames = []
    notes = []

    # 帧 0：全画布不透明红底，处置 1，延时 10。
    frames.append(
        frame_spec(0, 0, size, size, _rect_indices(size, size, 2),
                   delay_centis=10, disposal=1)
    )
    notes.append("全画布红底，处置1")

    # 帧 1：左上 4x4 绿色局部图，含透明像素（棋盘），验证透明叠层不覆盖旧红。
    w = h = 4
    idx = []
    for y in range(h):
        for x in range(w):
            idx.append(TRANSPARENT if (x + y) % 2 == 0 else 3)
    frames.append(
        frame_spec(0, 0, w, h, idx, transparent_index=TRANSPARENT,
                   delay_centis=5, disposal=1)
    )
    notes.append("左上4x4棋盘：绿/透明叠在红底上，处置1")

    # 帧 2：局部调色板，索引 3 在这里是青色而非全局绿色——同索引不同 RGB。
    local_palette = list(GLOBAL_PALETTE)
    local_palette[3] = (0, 255, 255)  # 青
    frames.append(
        frame_spec(4, 4, w, h, _rect_indices(w, h, 3),
                   palette=local_palette, use_local_palette=True,
                   delay_centis=7, disposal=3)
    )
    notes.append("右下4x4局部调色板：索引3=青（全局为绿），处置3")

    # 帧 3：连续第二个处置 3，恢复到帧 2 绘制前画布（即帧 1 的结果）。
    frames.append(
        frame_spec(2, 2, 3, 3, _rect_indices(3, 3, 5),  # 黄块
                   transparent_index=TRANSPARENT,
                   delay_centis=6, disposal=3)
    )
    notes.append("(2,2)3x3黄块，连续第二个处置3，恢复到帧2绘制前")

    # 帧 4：处置 2 的非整画布子矩形，绘制后只把该子矩形清回背景色。
    frames.append(
        frame_spec(1, 5, 5, 2, _rect_indices(5, 2, 4),  # 蓝条
                   delay_centis=8, disposal=2)
    )
    notes.append("(1,5)5x2蓝条，处置2：仅该子矩形恢复背景黑")

    # 帧 5：0 延时（按 1 处理），处置 0（合同按 1 处理）。
    frames.append(
        frame_spec(5, 1, 2, 2, _rect_indices(2, 2, 6),  # 橙块
                   delay_centis=0, disposal=0)
    )
    notes.append("(5,1)2x2橙块，0延时按1处理，处置0按1处理")

    # 帧 6：局部调色板 + 透明，验证局部表不改变其他区域已有颜色。
    local_palette2 = list(GLOBAL_PALETTE)
    local_palette2[2] = (200, 0, 200)  # 局部索引 2 变紫
    idx6 = []
    for y in range(4):
        for x in range(4):
            idx6.append(TRANSPARENT if x == 0 else 2)
    frames.append(
        frame_spec(0, 0, 4, 4, idx6, palette=local_palette2,
                   use_local_palette=True, transparent_index=TRANSPARENT,
                   delay_centis=12, disposal=1)
    )
    notes.append("左上4x4局部表：索引2=紫，首列透明，验证不影响区域外颜色")

    write_gif(path, size, size, bg_index, GLOBAL_PALETTE, frames)
    return {"path": path, "size": size, "frames": notes}


def build_scale_gif(
    path: str,
    width: int = 512,
    height: int = 512,
    frame_count: int = 1000,
    seed: int = 20261118,
) -> dict:
    """大规模输入：随机子矩形、全局/局部调色板、透明、四种处置、含 0 延时。"""
    rng = random.Random(seed)
    palette = [
        (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for _ in range(8)
    ]
    frames = []
    for i in range(frame_count):
        w = rng.randrange(8, 65)
        h = rng.randrange(8, 65)
        left = rng.randrange(0, width - w + 1)
        top = rng.randrange(0, height - h + 1)
        color = rng.randrange(0, 7)
        use_local = (i % 5 == 0)
        transparent_index = 7 if i % 3 == 0 else None
        if transparent_index is None:
            indices = _rect_indices(w, h, color)
        else:
            # 约 1/4 像素透明。
            indices = [
                TRANSPARENT if rng.random() < 0.25 else color
                for _ in range(w * h)
            ]
        local_palette = None
        if use_local:
            local_palette = list(palette)
            # 局部调色板故意改一个同索引不同 RGB 的条目。
            local_palette[color] = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        frames.append(
            frame_spec(
                left, top, w, h, indices,
                palette=local_palette,
                use_local_palette=use_local,
                transparent_index=transparent_index,
                delay_centis=rng.choice([0, 1, 2, 5, 10, 20]),
                disposal=rng.choice([0, 1, 2, 3]),
            )
        )
    write_gif(path, width, height, background_index=0,
              global_palette=palette, frames=frames)
    return {"path": path, "width": width, "height": height, "frame_count": frame_count}
