"""最小 GIF89a 容器读写器（只处理本合同支持的特性）。

支持：全局/局部颜色表、透明色索引、图像子矩形、帧延时、处置方式 0/1/2/3。
不支持也不生成：交错图像、纯文本扩展、网络循环扩展（动画只播放一遍）。

读取侧只做 *结构解析*：逐块返回索引像素、调色板与元数据，
绝不执行任何画布合成，因此不会把第三方库的默认展示行为当作真值。
LZW 由本仓库 lzw 模块独立完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import lzw

GIF_TRAILER = 0x3B
EXTENSION_INTRODUCER = 0x21
IMAGE_DESCRIPTOR = 0x2C
GRAPHIC_CONTROL_LABEL = 0xF9
APPLICATION_EXTENSION_LABEL = 0xFF
COMMENT_EXTENSION_LABEL = 0xFE


@dataclass(frozen=True)
class GifImage:
    """结构解析得到的单帧原始信息（无合成）。"""

    index: int
    left: int
    top: int
    width: int
    height: int
    indices: bytes  # 调色板索引，长度 width*height
    palette: tuple[tuple[int, int, int], ...]  # 本帧实际生效的 RGB 调色板
    palette_is_local: bool
    transparent_index: int | None  # GCE 声明的透明色索引
    delay_centis: int  # 百分之一秒单位的原始延时（可能为 0）
    disposal: int  # 原始处置方式 0..3（0 未归一化）


@dataclass(frozen=True)
class GifFile:
    width: int
    height: int
    background_index: int
    global_palette: tuple[tuple[int, int, int], ...]
    images: tuple[GifImage, ...]


@dataclass
class _FrameSpec:
    left: int
    top: int
    width: int
    height: int
    indices: list[int]
    palette: list[tuple[int, int, int]]
    use_local_palette: bool
    transparent_index: int | None
    delay_centis: int
    disposal: int


# --------------------------------------------------------------------------- #
# 写入
# --------------------------------------------------------------------------- #

def _color_table(palette: list[tuple[int, int, int]]) -> bytes:
    size = len(palette)
    if not 2 <= size <= 256 or (size & (size - 1)) != 0:
        raise ValueError("调色板条目数必须是 2..256 之间的 2 的幂")
    out = bytearray()
    for r, g, b in palette:
        out.extend((r & 0xFF, g & 0xFF, b & 0xFF))
    # 颜色表按 2 的幂补齐。
    out.extend(b"\x00" * ((size if size else 2) * 3 - len(out)))
    return bytes(out)


def _subblocks(data: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(data), 255):
        chunk = data[i : i + 255]
        out.append(len(chunk))
        out.extend(chunk)
    out.append(0)
    return bytes(out)


def _wrap_subblocks(payload: bytes) -> bytes:
    """给已剥离长度前缀的 LZW 净荷重新包子块，供 lzw.decode 消费。"""
    return _subblocks(payload)


def _gce(delay_centis: int, disposal: int, transparent_index: int | None) -> bytes:
    if disposal not in (0, 1, 2, 3):
        raise ValueError("处置方式只支持 0/1/2/3")
    packed = (disposal & 0b111) << 3
    if transparent_index is not None:
        packed |= 1
    return bytes(
        (
            EXTENSION_INTRODUCER,
            GRAPHIC_CONTROL_LABEL,
            4,
            packed,
            delay_centis & 0xFF,
            (delay_centis >> 8) & 0xFF,
            transparent_index if transparent_index is not None else 0,
            0,
        )
    )


def write_gif(
    path: str,
    width: int,
    height: int,
    background_index: int,
    global_palette: list[tuple[int, int, int]],
    frames: list[_FrameSpec],
) -> None:
    """按 GIF89a 写出单遍动画（无 NETSCAPE 循环扩展）。"""
    if width <= 0 or height <= 0 or width > 0xFFFF or height > 0xFFFF:
        raise ValueError("画布尺寸非法")
    if not global_palette:
        raise ValueError("全局调色板不能为空")
    gct = _color_table(global_palette)
    # GIF 颜色表大小位：条目数 n = 2**(bit+1)，故 bit = log2(n)-1。
    gct_size_bit = (len(gct) // 3).bit_length() - 2
    if not 0 <= background_index < len(global_palette):
        raise ValueError("背景索引超出全局调色板范围")

    out = bytearray(b"GIF89a")
    out.extend(
        (
            width & 0xFF,
            (width >> 8) & 0xFF,
            height & 0xFF,
            (height >> 8) & 0xFF,
            0x80 | (gct_size_bit << 4) | gct_size_bit,  # GCT 标志 + 色深 + 表大小
            background_index,
            0,  # 像素宽高比
        )
    )
    out.extend(gct)

    for fr in frames:
        if fr.use_local_palette:
            lct = _color_table(fr.palette)
            lct_bit = (len(lct) // 3).bit_length() - 2
        else:
            lct = b""
            lct_bit = 0
        if not (0 <= fr.left and fr.left + fr.width <= width) or not (
            0 <= fr.top and fr.top + fr.height <= height
        ):
            raise ValueError("帧子矩形越界")
        if len(fr.indices) != fr.width * fr.height:
            raise ValueError("索引数量与子矩形面积不符")
        if fr.transparent_index is not None and not 0 <= fr.transparent_index < (
            len(fr.palette) if fr.use_local_palette else len(global_palette)
        ):
            raise ValueError("透明索引超出实际调色板范围")

        out.extend(_gce(fr.delay_centis, fr.disposal, fr.transparent_index))
        out.append(IMAGE_DESCRIPTOR)
        out.extend(
            (
                fr.left & 0xFF,
                (fr.left >> 8) & 0xFF,
                fr.top & 0xFF,
                (fr.top >> 8) & 0xFF,
                fr.width & 0xFF,
                (fr.width >> 8) & 0xFF,
                fr.height & 0xFF,
                (fr.height >> 8) & 0xFF,
                (0x80 | lct_bit) if fr.use_local_palette else 0,
            )
        )
        if fr.use_local_palette:
            out.extend(lct)
        palette_len = len(fr.palette) if fr.use_local_palette else len(global_palette)
        min_code_size = max(2, (palette_len - 1).bit_length())
        out.extend(lzw.encode(fr.indices, min_code_size))

    out.append(GIF_TRAILER)
    with open(path, "wb") as fh:
        fh.write(out)


def frame_spec(
    left: int,
    top: int,
    width: int,
    height: int,
    indices: list[int],
    palette: list[tuple[int, int, int]] | None = None,
    use_local_palette: bool = False,
    transparent_index: int | None = None,
    delay_centis: int = 10,
    disposal: int = 1,
) -> _FrameSpec:
    """构造 write_gif 的帧描述（palette 为 None 时沿用全局调色板）。"""
    return _FrameSpec(
        left=left,
        top=top,
        width=width,
        height=height,
        indices=list(indices),
        palette=palette or [],
        use_local_palette=use_local_palette,
        transparent_index=transparent_index,
        delay_centis=delay_centis,
        disposal=disposal,
    )


# --------------------------------------------------------------------------- #
# 结构解析（不做合成）
# --------------------------------------------------------------------------- #

def _read_color_table(buf: bytes, pos: int, size_bit: int) -> tuple[tuple[tuple[int, int, int], ...], int]:
    count = 1 << (size_bit + 1)
    end = pos + count * 3
    table = tuple(
        (buf[pos + 3 * i], buf[pos + 3 * i + 1], buf[pos + 3 * i + 2])
        for i in range(count)
    )
    return table, end


def _skip_extension_blocks(buf: bytes, pos: int) -> int:
    """跳过扩展（含应用/注释扩展的数据子块），返回块结束符之后的位置。"""
    while pos < len(buf):
        size = buf[pos]
        pos += 1
        if size == 0:
            return pos
        pos += size
    raise ValueError("扩展块缺少结束符")


def parse_gif(path: str) -> GifFile:
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError("不是 GIF 文件")
    pos = 6
    width = buf[pos] | (buf[pos + 1] << 8)
    height = buf[pos + 2] | (buf[pos + 3] << 8)
    logical = buf[pos + 4]
    background_index = buf[pos + 5]
    pos += 7
    has_gct = bool(logical & 0x80)
    gct_bit = logical & 0b111
    if has_gct:
        global_palette, pos = _read_color_table(buf, pos, gct_bit)
    else:
        global_palette = ()

    images: list[GifImage] = []
    pending_delay = 0
    pending_disposal = 0
    pending_transparent: int | None = None

    while pos < len(buf):
        marker = buf[pos]
        pos += 1
        if marker == GIF_TRAILER:
            break
        if marker == EXTENSION_INTRODUCER:
            label = buf[pos]
            pos += 1
            if label == GRAPHIC_CONTROL_LABEL:
                # 此时 pos 指向 block_size 字节。
                block_size = buf[pos]
                if block_size != 4:
                    raise ValueError("GCE 块长度非法")
                packed = buf[pos + 1]
                pending_delay = buf[pos + 2] | (buf[pos + 3] << 8)
                pending_disposal = (packed >> 3) & 0b111
                t_flag = packed & 1
                ti = buf[pos + 4]
                pending_transparent = ti if t_flag else None
                # 1 字节块大小 + 4 字节内容 + 1 字节块结束符。
                pos += 6
                if buf[pos - 1] != 0:
                    raise ValueError("GCE 缺少块结束符")
            else:
                # 注释扩展、应用扩展等：本合同不生成，遇到则整体跳过。
                pos = _skip_extension_blocks(buf, pos)
            continue
        if marker == IMAGE_DESCRIPTOR:
            left = buf[pos] | (buf[pos + 1] << 8)
            top = buf[pos + 2] | (buf[pos + 3] << 8)
            iw = buf[pos + 4] | (buf[pos + 5] << 8)
            ih = buf[pos + 6] | (buf[pos + 7] << 8)
            image_packed = buf[pos + 8]
            pos += 9
            local_palette: tuple[tuple[int, int, int], ...] = ()
            if image_packed & 0x80:
                if image_packed & 0x40:
                    raise ValueError("交错图像不在本合同支持范围内")
                local_palette, pos = _read_color_table(buf, pos, image_packed & 0b111)
            min_code_size = buf[pos]
            pos += 1
            payload = bytearray()
            while True:
                size = buf[pos]
                pos += 1
                if size == 0:
                    break
                payload.extend(buf[pos : pos + size])
                pos += size
            indices = lzw.decode(bytes((min_code_size,)) + _wrap_subblocks(bytes(payload)))
            if len(indices) != iw * ih:
                raise ValueError("解码像素数与子矩形面积不符")
            palette = local_palette if local_palette else global_palette
            images.append(
                GifImage(
                    index=len(images),
                    left=left,
                    top=top,
                    width=iw,
                    height=ih,
                    indices=bytes(indices),
                    palette=palette,
                    palette_is_local=bool(local_palette),
                    transparent_index=pending_transparent,
                    delay_centis=pending_delay,
                    disposal=pending_disposal,
                )
            )
            pending_delay = 0
            pending_disposal = 0
            pending_transparent = None
            continue
        raise ValueError(f"遇到未知块标记 0x{marker:02X}（位置 {pos - 1}）")

    return GifFile(
        width=width,
        height=height,
        background_index=background_index,
        global_palette=tuple(global_palette),
        images=tuple(images),
    )
