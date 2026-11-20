# 可编辑动画帧目录服务与真实显示帧数据集生产器

本项目不把 GIF 库的默认展示行为当作真值。GIF 的 **LZW 编解码**（`lzw.py`）与
**容器结构解析**（`gif_io.py`）均独立实现并经 Pillow 交叉验证；合成口径在本仓库
合同内显式定义，由纯 Python 二维数组参考实现逐像素核对。

## 模块分层（依赖只允许向下）

| 模块 | 职责 |
| --- | --- |
| `animation_pipeline/lzw.py` | 独立 GIF LZW 可变长码字编解码（LSB 位打包、字典满 clear、KwKwK、流末尾 EOI 容错） |
| `animation_pipeline/gif_io.py` | GIF89a 容器写出与**纯结构解析**（不做任何画布合成） |
| `animation_pipeline/catalog.py` | 可编辑帧目录服务（稳定帧身份、边界/索引校验、LRU 正文缓存、持久化、原始帧 API） |
| `animation_pipeline/compositor.py` | 显示画布合成、处置 0/1/2/3、检查点、有理数时间选择、数据集流式生产 |
| `animation_pipeline/generator.py` | 自行生成真实 GIF 测试输入（小而可手算的特性集 + 512×512/1000 帧规模输入） |
| `animation_pipeline/demo.py` | 端到端演示 |

`compositor.py` 不导入 `gif_io`，也不打开原 GIF；它只消费目录的
`iter_raw_frames()/get_frame()` 原始帧 API。数据集生产器因此无法绕过目录
直接取库生成的最终帧。

## 合成合同口径

* 初始画布为逻辑背景色的完全不透明 RGBA。
* 处置 0 按处置 1 处理：局部 `alpha>0` 像素覆盖当前画布，透明像素不覆盖旧像素；
  局部调色板只影响本帧子矩形，不改变其他区域已存在的颜色。
* 每帧**先保存绘制完成的完整显示画布**，再按处置准备下一帧：
  * 2：把该帧子矩形恢复为逻辑背景色（不透明）；
  * 3：恢复到绘制本帧之前保存的画布（支持连续多个处置 3）。
* 延时单位百分之一秒，0 延时按 1 处理。
* 半开区间 `[start, end)` 选帧，恰在终点选下一帧；只播放一遍，末尾之后返回结束，不循环。

## 目录产品（`animgif-catalog` v1）

```
product_dir/
  catalog.json          # 画布、背景RGB、每帧矩形/延时/处置/调色板引用/正文路径
  bodies/fid_<id>.png   # 帧 RGBA 正文（真实 PNG）
  bodies/fid_<id>.idx   # GIF 来源索引图（供调色板修订重渲染，可选）
```

* `frame_id` 稳定，删除后永不复用；编辑使 `revision` 递增。
* 改矩形验证画布边界与正文形状；改调色板验证索引范围并重渲染 RGBA。
* 正文以 PNG 落盘，内存仅保留字节预算内的 LRU（默认 12 MiB，硬上限 16 MiB）。
* `publish_to()` 可在修订后把完整产品发布到新目录，脱离旧数据集独立迭代。

## 数据集产品（`animgif-dataset` v1）

```
output_dir/
  manifest.json         # 画布、背景、样本数、唯一 PNG 数、结束时间、不循环
  samples.jsonl         # 每采样一行：时间、frame_id、RGBA 摘要、覆盖/继承/透明计数
  frames/<sha256前16>.png  # 内容寻址的真实显示画布 PNG
```

* 时间点接受 `int/Fraction/浮点/分数字符串`，统一为精确有理数。
* 同一显示画布（RGBA 摘要相同）的多个采样共享一个 PNG，但 JSONL 中保留独立样本身份。
* 结束样本 `ended=true`、不带 PNG/frame_id。
* 所有覆盖/继承/透明计数均可仅凭目录原始帧用 `reference_replay` 重算。

## 运行

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m animation_pipeline.demo
.venv/bin/python -m unittest discover -s tests -v
```

## 测试覆盖（50 个用例）

* LZW：全 mcs(2–8) 往返、字典满 clear、KwKwK、256/4096 码边界 EOI、截断拒绝、Pillow 外部交叉。
* GIF 容器：全局/局部调色板同索引不同 RGB、透明索引、子矩形、四种处置逐字段保真、交错拒绝。
* 合成：透明叠层、处置 0→1、处置 2 非整画布清背景、连续处置 3、0 延时、半开区间、不循环、检查点。
* 目录：身份稳定不复用、矩形越界/索引越界校验、调色板修订重渲染、持久化往返、发布新产品。
* 数据集：真实 PNG 回读验 RGBA、计数可由原始帧重算、共享 PNG 不丢样本身份、结束态、乱序保身份、编辑重算、架构隔离。
* 规模：512×512、1000 帧、10000 采样，正文内存峰值 < 16 MiB，流式落盘。
