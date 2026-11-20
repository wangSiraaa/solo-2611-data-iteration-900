"""GIF 容器写/结构解析测试：元数据逐字段保真，解析侧不做任何合成。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from animation_pipeline.gif_io import frame_spec, parse_gif, write_gif

PALETTE = [
    (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0),
    (0, 0, 255), (255, 255, 0), (255, 128, 0), (128, 128, 128),
]
T = 7


class GifContainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "x.gif")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, frames):
        write_gif(self.path, 8, 8, 0, PALETTE, frames)

    def test_basic_metadata_roundtrip(self) -> None:
        frames = [
            frame_spec(1, 2, 3, 4, [2] * 12, delay_centis=15, disposal=2),
            frame_spec(0, 0, 2, 2, [T, 1, 1, T], transparent_index=T,
                       delay_centis=0, disposal=3),
        ]
        self._write(frames)
        gif = parse_gif(self.path)
        self.assertEqual((gif.width, gif.height), (8, 8))
        self.assertEqual(gif.background_index, 0)
        self.assertEqual(len(gif.images), 2)

        f0 = gif.images[0]
        self.assertEqual((f0.left, f0.top, f0.width, f0.height), (1, 2, 3, 4))
        self.assertEqual(bytes(f0.indices), b"\x02" * 12)
        self.assertEqual(f0.delay_centis, 15)
        self.assertEqual(f0.disposal, 2)
        self.assertIsNone(f0.transparent_index)
        self.assertFalse(f0.palette_is_local)

        f1 = gif.images[1]
        self.assertEqual(f1.delay_centis, 0)  # 0 延时原样保留，归一化在合成侧
        self.assertEqual(f1.disposal, 3)
        self.assertEqual(f1.transparent_index, T)
        self.assertEqual(bytes(f1.indices), bytes((T, 1, 1, T)))

    def test_local_palette_same_index_different_rgb(self) -> None:
        local = list(PALETTE)
        local[3] = (0, 255, 255)  # 全局索引 3 是绿，局部改青
        self._write([
            frame_spec(0, 0, 2, 2, [3] * 4, delay_centis=10, disposal=1),
            frame_spec(2, 2, 2, 2, [3] * 4, palette=local,
                       use_local_palette=True, delay_centis=10, disposal=1),
        ])
        gif = parse_gif(self.path)
        self.assertFalse(gif.images[0].palette_is_local)
        self.assertEqual(gif.images[0].palette[3], (0, 255, 0))
        self.assertTrue(gif.images[1].palette_is_local)
        self.assertEqual(gif.images[1].palette[3], (0, 255, 255))
        # 两份索引序列相同，但调色板解析出不同 RGB——结构信息完整保留。
        self.assertEqual(gif.images[0].indices, gif.images[1].indices)

    def test_all_disposals_preserved(self) -> None:
        self._write([
            frame_spec(0, 0, 1, 1, [1], delay_centis=5, disposal=d)
            for d in (0, 1, 2, 3)
        ])
        gif = parse_gif(self.path)
        self.assertEqual([im.disposal for im in gif.images], [0, 1, 2, 3])

    def test_parser_returns_raw_indices_not_canvas(self) -> None:
        # 子矩形帧的索引数组尺寸必须等于子矩形面积，而非整画布——证明无合成。
        self._write([frame_spec(3, 3, 2, 2, [1, 2, 3, 4], disposal=1)])
        gif = parse_gif(self.path)
        im = gif.images[0]
        self.assertEqual(len(im.indices), 4)
        self.assertEqual((im.width, im.height), (2, 2))

    def test_rejects_interlaced_marker_if_present(self) -> None:
        # 手工构造带交错标志的图像描述符，解析器应明确拒绝。
        body = bytearray(b"GIF89a")
        body += (8).to_bytes(2, "little") + (8).to_bytes(2, "little")
        body += bytes((0x80 | (2 << 4) | 2, 0, 0))
        body += bytes(sum((list(c) for c in PALETTE), []))
        body += b"\x2C" + (0).to_bytes(4, "little") + (8).to_bytes(2, "little") + (8).to_bytes(2, "little")
        body += bytes((0x40,))  # interlace 置位
        body += b"\x02\x02\x44\x01\x00\x3B"
        p = str(Path(self.tmp.name) / "interlaced.gif")
        Path(p).write_bytes(bytes(body))
        with self.assertRaises(ValueError):
            parse_gif(p)

    def test_writer_rejects_out_of_bounds_rect(self) -> None:
        with self.assertRaises(ValueError):
            self._write([frame_spec(7, 7, 2, 2, [1] * 4, disposal=1)])


if __name__ == "__main__":
    unittest.main()
