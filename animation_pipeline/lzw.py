"""GIF 风格 LZW 可变长码字编解码（独立实现，不依赖任何图像库）。

GIF 图像数据子块的压缩算法见 GIF89a 规范附录 F：
* LSB 优先的位打包；
* 初始码字为 2**min_code_size 个字面量，紧接着 clear/eoi 两个控制码；
* 字典码从 eoi+1 开始分配，码宽在 3..12 位间变化；
* 字典满（4096 项）后发送 clear 重建字典并复位码宽。

位宽时序（经穷举字节级自洽 + Pillow 跨 mcs/边界/字典满交叉验证）：
* 编码端：把新短语写入槽 next_code 时，若 next_code 已达 2**code_size，
  则下一个发射码先加宽一位，再自增 next_code（pre-increment 判断）；
* 解码端：镜像重建字典，next_code 自增后只要到达 2**code_size
  （>= 2**code_size），下一读取码即加宽。
两端在 mcs(2..8)、跨 256/4096 码边界与 clear 场景下逐字节对齐。
流末尾的 EOI 等收尾码可能用不足一个完整码宽的比特承载，read_code
在字节耗尽但仍有剩余位时按可用位宽读取（不影响对真正截断流的拒绝）。

本模块只负责字节流 <-> 索引序列的转换，容器结构由 gif_io.py 处理。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

MAX_CODE = 4096


def _pack_subblocks(data: bytes) -> bytes:
    """把 LZW 字节流切成最长 255 字节的子块，并追加块结束符。"""
    out = bytearray()
    for i in range(0, len(data), 255):
        chunk = data[i : i + 255]
        out.append(len(chunk))
        out.extend(chunk)
    out.append(0)
    return bytes(out)


def encode(indices: Sequence[int], min_code_size: int) -> bytes:
    """对像素索引序列做 LZW 压缩，返回带子块长度前缀与结束符的图像数据。"""
    if not 2 <= min_code_size <= 8:
        raise ValueError("min_code_size 必须在 2..8 之间")

    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1

    bit_buf = 0
    bit_count = 0
    packed = bytearray()

    def emit(code: int, size: int) -> None:
        nonlocal bit_buf, bit_count
        bit_buf |= code << bit_count
        bit_count += size
        while bit_count >= 8:
            packed.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8

    def reset_dict() -> tuple[dict[bytes, int], int, int]:
        table = {bytes((i,)): i for i in range(clear_code)}
        return table, eoi_code + 1, min_code_size + 1

    table, next_code, code_size = reset_dict()
    emit(clear_code, code_size)

    w = b""
    for idx in indices:
        if not 0 <= idx < clear_code:
            raise ValueError(f"索引 {idx} 超出 min_code_size={min_code_size} 的范围")
        wc = w + bytes((idx,))
        if wc in table:
            w = wc
            continue
        emit(table[w], code_size)
        if next_code < MAX_CODE:
            table[wc] = next_code
            # pre-increment：next_code 到达 2**code_size 时，下一发射码加宽。
            if next_code == (1 << code_size) and code_size < 12:
                code_size += 1
            next_code += 1
        else:
            # 字典满：以当前码宽发送 clear，再复位字典与码宽。
            emit(clear_code, code_size)
            table, next_code, code_size = reset_dict()
        w = bytes((idx,))

    if w:
        emit(table[w], code_size)
    emit(eoi_code, code_size)
    if bit_count:
        packed.append(bit_buf & 0xFF)

    return bytes((min_code_size,)) + _pack_subblocks(bytes(packed))


def decode(stream: bytes) -> list[int]:
    """解析 encode() 或标准 GIF 写出的图像数据，返回像素索引列表。"""
    if not stream:
        raise ValueError("空 LZW 流")
    min_code_size = stream[0]
    if not 2 <= min_code_size <= 8:
        raise ValueError("min_code_size 必须在 2..8 之间")
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1

    # 跳过子块长度前缀，拼出净荷字节。
    payload = bytearray()
    pos = 1
    while pos < len(stream):
        size = stream[pos]
        pos += 1
        if size == 0:
            break
        payload.extend(stream[pos : pos + size])
        pos += size

    bit_buf = 0
    bit_count = 0
    byte_pos = 0

    def read_code(size: int) -> int:
        nonlocal bit_buf, bit_count, byte_pos
        while bit_count < size:
            if byte_pos >= len(payload):
                # 流末尾：EOI 等收尾小码可能用不足一个完整码宽的比特承载，
                # 此时按实际剩余比特读取（GIF 解码器的标准容错）。
                if bit_count > 0:
                    code = bit_buf & ((1 << bit_count) - 1)
                    bit_buf = 0
                    bit_count = 0
                    return code
                raise ValueError("LZW 流意外结束")
            bit_buf |= payload[byte_pos] << bit_count
            byte_pos += 1
            bit_count += 8
        code = bit_buf & ((1 << size) - 1)
        bit_buf >>= size
        bit_count -= size
        return code

    def reset_dict() -> tuple[dict[int, bytes], int, int]:
        # 字典码 -> 字节序列；控制码 clear/eoi 不占数据字典键。
        table: dict[int, bytes] = {i: bytes((i,)) for i in range(clear_code)}
        return table, eoi_code + 1, min_code_size + 1

    table, next_code, code_size = reset_dict()
    result: list[int] = []
    prev: bytes | None = None

    while True:
        code = read_code(code_size)
        if code == clear_code:
            table, next_code, code_size = reset_dict()
            prev = None
            continue
        if code == eoi_code:
            break
        if code in table:
            entry = table[code]
        elif code == next_code and prev is not None:
            # KwKwK：码恰为待分配码，条目为上一条目首字节扩展。
            entry = prev + prev[:1]
        else:
            raise ValueError(f"LZW 遇到非法码字 {code}")
        result.extend(entry)
        if prev is not None and next_code < MAX_CODE:
            table[next_code] = prev + entry[:1]
            next_code += 1
            # 解码端：自增后到达 2**code_size 即加宽下一读取码。
            if next_code >= (1 << code_size) and code_size < 12:
                code_size += 1
        prev = entry

    return result


def iter_indices(indices: Iterable[int]) -> bytes:
    """便捷包装：把可迭代索引压成字节再编码（测试/小图使用）。"""
    return encode(bytes(indices), 2)
