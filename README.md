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
| `animation_pipeline/filmstrip.py` | 片条产品：从目录切连续帧、任意合法入场底画回放、符号来源追溯、片条样本生成 |
| `animation_pipeline/cabinet.py` | 展柜饰面合同：旧柜独立色单 / 新柜漆罐批号库存（Dinic 最大流整柜可行性，不扣库存） |
| `animation_pipeline/substitution.py` | 替片校核单：免枚举全域校核、真实可装反例、签单持久化与改稿/库存变更后重开 |
| `animation_pipeline/stripbuilders.py` | 题面标准件片条/局部色表改稿素材 |
| `animation_pipeline/demo.py` | 端到端演示 |
| `animation_pipeline/strip_demo.py` | 片条回放与替片校核全部验收场景演示 |

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
.venv/bin/python -m animation_pipeline.strip_demo
.venv/bin/python -m unittest discover -s tests -v
```

## 片条产品（`animgif-filmstrip` v1）

片条是从帧目录切出的**固定可重用产品**（`Filmstrip.from_catalog` 只接受按目录顺序的
连续帧身份），记录原逻辑画布尺寸、该片目录的逻辑背景色与连续帧序列：

```
strip_dir/
  strip.json            # 画布、片条逻辑背景、帧矩形/延时/处置/局部色表引用、产品指纹
  bodies/fid_<id>.png   # 帧 RGBA 正文
```

* **不继承首帧之前的电影内容**：进入片条时每个像素携带调用方整张入场底画的颜色。
  `replay(entry_rgba)` / `exit_canvas(entry_rgba)` 接受任意与画布同形的合法入场底画。
* 合成口径与整片一致（透明不覆盖、处置 0 按 1、先存显示再处置、处置 3 恢复画前）；
  **处置 2 一律恢复片条自己的逻辑背景色**，不把它当作调用方入场色。
* `symbolic_trace()` 做一次与入场底画无关的**符号回放**：每个像素在任一显示时刻与
  离场底画上只属于两类来源项——`entry`（某坐标入场像素的延续）或 `constant`
  （确定 RGB，原因可定位到帧：`paint` / `restore_background`；处置 3 经
  `restore_before` 回到画前项）。`explain_pixel()` 给出单像素逐帧事件链。
* `produce_strip_samples()` 从任意入场底画（含逻辑背景入场）产出真实 PNG/JSONL；
  原整片从逻辑背景开始的 `produce_dataset` 保持不变。

## 展柜合同与整柜可行性（`cabinet.py`）

展柜把逻辑画布按坐标划分成**完整覆盖、互不重叠**的饰面区，整柜底画=每区恰好一色。

* **旧柜 `LegacyCabinet`**：无漆罐条款，每区从批准 RGB 清单独立选一色；清单非空即可配。
* **新柜 `PaintCanCabinet`**：漆按**批号**封装，同色可来自不同批号；每区列出兼容批号，
  一罐刷一区，同批号罐数有限（0 罐=缺货）。可行性由 `源→批号(容量=罐数)→区(兼容边)→汇`
  的 Dinic 最大流判定，**绝不按 RGB 并库**：两个同色批号若各自只适配部分区，分别计库存。
* 只把整柜能同时配齐的 `Scheme` 当合法底画；`supports_scheme` 只核对不实际扣库存；
  配不齐时 `any_scheme()` 抛 `CabinetUnsatisfiable`。

## 替片校核单（`animgif-substitution-sheet` v1）

`verify_substitution(甲片, 乙片, 展柜, 样本时刻, exit_regions=...)` 的判定口径：

* 两片有效总时长（0 延时按 1）必须相同，否则 `input_error`；
* 整柜根本配不齐 → `cabinet_unsatisfiable`，**不能空判替片成功**；
* **全程不枚举行入场底画**：对柜方每个样本时刻逐像素比较两片符号项——
  `entry/entry` 恒等、同 RGB 常量恒等；`entry/常量` 或异色常量只在存在一套真实可装
  整柜方案能暴露差异时才判失败；
* 失败报告含一套真实可装整柜方案（新柜逐区列批号）、最早不符的样本/坐标/两侧实际颜色
  与可定位原因；检查顺序为显示采样（样本序→行主序坐标）在前，**离场区检查排在最后**；
* `issue_sheet()` 落盘签单（含两片与柜约/库存指纹）；`reopen_sheet()` 复读旧结论、
  标出片条换版/柜约/库存的指纹漂移并按当前对象重新判定，支持上百饰面区与多场次复用。

## 测试覆盖（88 个用例）

* LZW：全 mcs(2–8) 往返、字典满 clear、KwKwK、256/4096 码边界 EOI、截断拒绝、Pillow 外部交叉。
* GIF 容器：全局/局部调色板同索引不同 RGB、透明索引、子矩形、四种处置逐字段保真、交错拒绝。
* 合成：透明叠层、处置 0→1、处置 2 非整画布清背景、连续处置 3、0 延时、半开区间、不循环、检查点。
* 目录：身份稳定不复用、矩形越界/索引越界校验、调色板修订重渲染、持久化往返、发布新产品。
* 数据集：真实 PNG 回读验 RGBA、计数可由原始帧重算、共享 PNG 不丢样本身份、结束态、乱序保身份、编辑重算、架构隔离。
* 规模：512×512、1000 帧、10000 采样，正文内存峰值 < 16 MiB，流式落盘。
* 片条：连续帧切出校验、任意入场底画回放、处置2 用片条背景、处置3 恢复画前、
  不继承片前内容、符号来源项与事件链、符号求值与具体回放逐像素交叉验证、持久化指纹。
* 展柜：分区覆盖/重叠校验、旧柜独立色单、新柜批号库存最大流、缺货无解、
  核对不扣库存、同 RGB 批号不并库、120 区不枚举的可行性判定。
* 替片：白/红饰面可换与拒绝并复现红/白差异、处置3/1 显示可换但离场不可换、
  局部色表改稿后旧签单复读改判、两区柜 1 罐/2 罐/缺货三态、反例逐区批号真实可装、
  库存调整重开指纹漂移、时长不一致输入错误。
