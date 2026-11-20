"""LZW 编解码测试：往返、字典满 clear、与独立参考及 Pillow 交叉验证。"""

from __future__ import annotations

import io
import random
import unittest

import numpy as np
from PIL import Image

from animation_pipeline import lzw


def _reference_decode(stream: bytes) -> list[int]:
    """独立书写的 GIF 规范参考解码器（post-increment 位宽约定）。"""
    mcs = stream[0]
    clear = 1 << mcs
    eoi = clear + 1
    payload = bytearray()
    pos = 1
    while stream[pos]:
        size = stream[pos]
        pos += 1
        payload.extend(stream[pos : pos + size])
        pos += size

    code_size = mcs + 1
    bit_buf = bit_count = byte_pos = 0

    def read() -> int:
        nonlocal bit_buf, bit_count, byte_pos
        while bit_count < code_size:
            bit_buf |= payload[byte_pos] << bit_count
            byte_pos += 1
            bit_count += 8
        c = bit_buf & ((1 << code_size) - 1)
        bit_buf >>= code_size
        bit_count -= code_size
        return c

    def fresh_table() -> list[bytes | None]:
        return [bytes((i,)) for i in range(clear)] + [None, None]

    table = fresh_table()
    nxt = eoi + 1
    prev = None
    out: list[int] = []
    while True:
        code = read()
        if code == clear:
            table, nxt, code_size = fresh_table(), eoi + 1, mcs + 1
            prev = None
            continue
        if code == eoi:
            return out
        if code < clear:
            entry = bytes((code,))
        elif code < len(table) and table[code] is not None:
            entry = table[code]  # type: ignore[assignment]
        elif code == nxt and prev is not None:
            entry = prev + prev[:1]
        else:
            raise ValueError(f"非法码字 {code}")
        out.extend(entry)
        if prev is not None and nxt < 4096:
            table.append(prev + entry[:1])
            nxt += 1
            if nxt == (1 << code_size) and code_size < 12:
                code_size += 1
        prev = entry


class LzwTests(unittest.TestCase):
    def test_roundtrip_all_sizes(self) -> None:
        rng = random.Random(2026)
        for mcs in (2, 3, 4, 5, 6, 7, 8):
            n = 1 << mcs
            for _ in range(25):
                seq = [rng.randrange(n) for _ in range(rng.randint(1, 2000))]
                with self.subTest(mcs=mcs):
                    self.assertEqual(lzw.decode(lzw.encode(seq, mcs)), seq)

    def test_dictionary_full_clear(self) -> None:
        # 长常数序列必然多次占满 4096 字典并触发 clear。
        for mcs in (2, 3, 8):
            for color in range(1 << mcs):
                seq = [color] * 9000
                self.assertEqual(lzw.decode(lzw.encode(seq, mcs)), seq)

    def test_matches_independent_reference(self) -> None:
        rng = random.Random(77)
        for mcs in (2, 3, 4, 8):
            n = 1 << mcs
            for _ in range(20):
                seq = [rng.randrange(n) for _ in range(rng.randint(1, 1500))]
                stream = lzw.encode(seq, mcs)
                self.assertEqual(_reference_decode(stream), seq)
                self.assertEqual(lzw.decode(stream), seq)

    def test_kwkw_edge_case(self) -> None:
        # 制造“码恰为下一个待分配码”的 KwKwK 情形。
        seq = [0, 1, 0, 1, 0, 1, 2, 2, 0, 1, 2]
        self.assertEqual(lzw.decode(lzw.encode(seq, 2)), seq)

    def test_stream_ending_exactly_at_code_size_boundary(self) -> None:
        # 回归：低熵序列使字典在结束时恰好到达 2**code_size（如 256），
        # EOI 仍须按当前有效位宽正确读取，不得误切更宽位宽导致截断。
        rng = random.Random(20261118)
        seq = [7 if rng.random() < 0.25 else 0 for _ in range(1591)]
        stream = lzw.encode(seq, 3)
        self.assertEqual(lzw.decode(stream), seq)
        # 多个同形态长度，覆盖结尾停在不同字典码数的情形。
        for length in (255, 256, 257, 511, 512, 513, 1024, 1590, 1591):
            self.assertEqual(lzw.decode(lzw.encode(seq[:length], 3)), seq[:length])

    def test_truncated_stream_still_rejected(self) -> None:
        # 真正截断（字节被砍掉）必须报错，不能被流末尾容错吞掉。
        stream = lzw.encode([0, 1, 2, 3] * 100, 2)
        with self.assertRaises(ValueError):
            lzw.decode(stream[:-4])

    def test_pillow_decodes_our_gif_lzw(self) -> None:
        rng = random.Random(5)
        for mcs in (2, 3, 4, 8):
            n = 1 << mcs
            w = h = 32
            seq = [rng.randrange(n) for _ in range(w * h)]
            pal_n = n
            palette = bytes(
                sum(([i % 256, (i * 7) % 256, (i * 13) % 256] for i in range(pal_n)), [])
            )
            size_bit = pal_n.bit_length() - 2
            gif = (
                b"GIF89a"
                + w.to_bytes(2, "little") + h.to_bytes(2, "little")
                + bytes((0x80 | (size_bit << 4) | size_bit, 0, 0))
                + palette
                + b"\x2C" + (0).to_bytes(4, "little")
                + w.to_bytes(2, "little") + h.to_bytes(2, "little")
                + b"\x00" + lzw.encode(seq, mcs) + b"\x3B"
            )
            with Image.open(io.BytesIO(gif)) as im:
                got = list(im.get_flattened_data())
            self.assertEqual(got, seq)

    def test_rejects_out_of_range_index(self) -> None:
        with self.assertRaises(ValueError):
            lzw.encode([0, 1, 9], 2)


if __name__ == "__main__":
    unittest.main()
