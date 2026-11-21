"""放映柜饰面合同：旧柜独立色单与新柜成套漆罐批号。

展柜把逻辑画布按坐标划成若干饰面区，片条回放就跑在“整柜底画”上——
每个区由恰好一种颜色填满。

* 旧柜（LegacyCabinet）：每区从批准的 RGB 清单里选一种颜色，区与区独立。
* 新柜（PaintCanCabinet）：同色油漆可能来自不同*批号*；每区只列出兼容的批号，
  一罐恰好够刷一个区，同批可用罐数有限。判定不按 RGB 并库——两个同色批号若各自
  只适配部分区，必须分别计库存。

只把*整柜能同时配齐*的方案当作合法底画。可行性由源点→批号→饰面区→汇点的
最大流判定（每个区需求 1，批号容量=可用罐数，边表示批号对该区兼容），
核对方案只做容量检查，不实际扣减库存。本模块不含片条逻辑，可独立使用。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

RGB = tuple[int, int, int]


class CabinetValidationError(ValueError):
    """展柜区域/漆罐合同非法时抛出。"""


class CabinetUnsatisfiable(Exception):
    """整柜根本配不齐：柜约无解。"""


@dataclass(frozen=True)
class Region:
    """一个饰面区：矩形坐标（左,上,宽,高）+ 该区可选/兼容的颜色或批号。"""

    name: str
    left: int
    top: int
    width: int
    height: int

    def contains(self, x: int, y: int) -> bool:
        return (
            self.left <= x < self.left + self.width
            and self.top <= y < self.top + self.height
        )


@dataclass(frozen=True)
class PaintLot:
    """漆罐批号：一个确定 RGB，库存=可用罐数（0 表示该批号缺货）。"""

    lot_id: str
    rgb: RGB
    cans: int


@dataclass(frozen=True)
class RegionPaint:
    """单区的整柜装配：用哪个批号刷成什么颜色。"""

    region: str
    lot_id: str
    rgb: RGB


@dataclass(frozen=True)
class Scheme:
    """一套真实可装的整柜底画。

    colors[区名] = 填满该区的 RGB；
    paints 仅新柜非空，逐区给出批号，可直接照单领罐。
    """

    colors: dict[str, RGB]
    paints: tuple[RegionPaint, ...] = field(default_factory=tuple)

    def region_color(self, name: str) -> RGB:
        return self.colors[name]


# --------------------------------------------------------------------------- #
# Dinic 最大流（纯标准库）
# --------------------------------------------------------------------------- #

class _Dinic:
    def __init__(self, n: int) -> None:
        self.n = n
        self.graph: list[list[dict]] = [[] for _ in range(n)]
        self.level = [-1] * n

    def add_edge(self, u: int, v: int, capacity: int) -> None:
        forward = {"to": v, "cap": capacity, "rev": len(self.graph[v])}
        backward = {"to": u, "cap": 0, "rev": len(self.graph[u])}
        self.graph[u].append(forward)
        self.graph[v].append(backward)

    def _bfs(self, source: int, sink: int) -> bool:
        self.level = [-1] * self.n
        self.level[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                if edge["cap"] > 0 and self.level[edge["to"]] < 0:
                    self.level[edge["to"]] = self.level[u] + 1
                    queue.append(edge["to"])
        return self.level[sink] >= 0

    def _dfs(self, u: int, sink: int, pushed: int) -> int:
        if u == sink:
            return pushed
        for edge in self.graph[u]:
            if edge["cap"] > 0 and self.level[edge["to"]] == self.level[u] + 1:
                flow = self._dfs(edge["to"], sink, min(pushed, edge["cap"]))
                if flow > 0:
                    edge["cap"] -= flow
                    self.graph[edge["to"]][edge["rev"]]["cap"] += flow
                    return flow
        return 0

    def max_flow(self, source: int, sink: int) -> int:
        total = 0
        inf = 10**9
        while self._bfs(source, sink):
            while True:
                pushed = self._dfs(source, sink, inf)
                if pushed == 0:
                    break
                total += pushed
        return total


def _validate_partition(regions: list[Region], width: int, height: int) -> None:
    if not regions:
        raise CabinetValidationError("展柜至少需要一个饰面区")
    names = [r.name for r in regions]
    if len(set(names)) != len(names):
        raise CabinetValidationError("饰面区名称必须唯一")
    covered = [[False] * width for _ in range(height)]
    for r in regions:
        if r.width <= 0 or r.height <= 0:
            raise CabinetValidationError(f"区 {r.name} 宽高必须为正")
        if r.left < 0 or r.top < 0 or r.left + r.width > width or r.top + r.height > height:
            raise CabinetValidationError(f"区 {r.name} 越界（画布 {width}x{height}）")
        for y in range(r.top, r.top + r.height):
            for x in range(r.left, r.left + r.width):
                if covered[y][x]:
                    raise CabinetValidationError(f"饰面区在坐标 ({x},{y}) 重叠")
                covered[y][x] = True
    if any(not row[i] for row in covered for i in range(width)):
        raise CabinetValidationError("饰面区必须完整覆盖逻辑画布，不能留空")


class _CabinetBase:
    def __init__(self, width: int, height: int, regions: list[Region]) -> None:
        if width <= 0 or height <= 0:
            raise CabinetValidationError("画布宽高必须为正整数")
        _validate_partition(regions, width, height)
        self.width = width
        self.height = height
        self.regions = list(regions)
        self._region_by_name = {r.name: r for r in regions}

    def region_at(self, x: int, y: int) -> str:
        for r in self.regions:
            if r.contains(x, y):
                return r.name
        raise CabinetValidationError(f"坐标 ({x},{y}) 不属于任何饰面区")

    def paint_canvas(self, scheme: Scheme) -> list[list[RGB]]:
        """把一套整柜方案落成逐像素 RGB 二维数组（给片条当入场底画用）。"""
        canvas: list[list[RGB]] = [
            [(0, 0, 0)] * self.width for _ in range(self.height)
        ]
        for r in self.regions:
            color = scheme.region_color(r.name)
            for y in range(r.top, r.top + r.height):
                for x in range(r.left, r.left + r.width):
                    canvas[y][x] = color
        return canvas

    # 子类实现
    def is_satisfiable(self) -> bool:
        raise NotImplementedError

    def any_scheme(self) -> Scheme:
        raise NotImplementedError

    def scheme_with_region_color(self, region_name: str, rgb: RGB) -> Scheme | None:
        """求一套真实可装、且指定区被指定颜色填满的整柜方案；不存在返回 None。"""
        raise NotImplementedError

    def supports_scheme(self, scheme: Scheme) -> bool:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 旧柜：每区独立从批准 RGB 清单选一种，无漆罐条款。
# --------------------------------------------------------------------------- #

class LegacyCabinet(_CabinetBase):
    def __init__(
        self,
        width: int,
        height: int,
        regions: list[Region],
        color_lists: dict[str, list[RGB]],
    ) -> None:
        super().__init__(width, height, regions)
        if set(color_lists) != {r.name for r in regions}:
            raise CabinetValidationError("每个饰面区都必须给出批准 RGB 清单")
        for name, colors in color_lists.items():
            if not colors or any(len(c) != 3 for c in colors):
                raise CabinetValidationError(f"区 {name} 的色单为空或不是 RGB 三元组")
        self.color_lists = {k: tuple(v) for k, v in color_lists.items()}

    def is_satisfiable(self) -> bool:
        # 区与区独立，每区清单非空即整柜可配。
        return all(self.color_lists[r.name] for r in self.regions)

    def any_scheme(self) -> Scheme:
        if not self.is_satisfiable():
            raise CabinetUnsatisfiable("旧柜存在无可用颜色的饰面区，柜约无解")
        return Scheme(colors={r.name: self.color_lists[r.name][0] for r in self.regions})

    def scheme_with_region_color(self, region_name: str, rgb: RGB) -> Scheme | None:
        if region_name not in self._region_by_name:
            raise CabinetValidationError(f"未知饰面区 {region_name}")
        if rgb not in self.color_lists[region_name]:
            return None
        colors = {r.name: self.color_lists[r.name][0] for r in self.regions}
        colors[region_name] = rgb
        return Scheme(colors=colors)

    def supports_scheme(self, scheme: Scheme) -> bool:
        for r in self.regions:
            if r.name not in scheme.colors:
                return False
            if scheme.colors[r.name] not in self.color_lists[r.name]:
                return False
        return True


# --------------------------------------------------------------------------- #
# 新柜：成套漆罐，批号分别计库存、逐区兼容，不能按颜色并库。
# --------------------------------------------------------------------------- #

class PaintCanCabinet(_CabinetBase):
    def __init__(
        self,
        width: int,
        height: int,
        regions: list[Region],
        lots: list[PaintLot],
        compatible_lots: dict[str, list[str]],
    ) -> None:
        super().__init__(width, height, regions)
        if not lots:
            raise CabinetValidationError("新柜必须提供漆罐批号")
        lot_ids = [lot.lot_id for lot in lots]
        if len(set(lot_ids)) != len(lot_ids):
            raise CabinetValidationError("漆罐批号必须唯一")
        self.lots = {lot.lot_id: lot for lot in lots}
        if any(lot.cans < 0 for lot in lots):
            raise CabinetValidationError("漆罐可用罐数不能为负（0 表示该批号缺货）")
        if set(compatible_lots) != {r.name for r in regions}:
            raise CabinetValidationError("每个饰面区都必须给出兼容批号清单")
        self.compatible_lots: dict[str, tuple[str, ...]] = {}
        for r in regions:
            ids = tuple(compatible_lots[r.name])
            if not ids:
                raise CabinetValidationError(f"区 {r.name} 没有任何兼容漆罐")
            if any(lot_id not in self.lots for lot_id in ids):
                raise CabinetValidationError(f"区 {r.name} 引用了不存在的批号")
            if len(set(ids)) != len(ids):
                raise CabinetValidationError(f"区 {r.name} 的兼容批号重复")
            self.compatible_lots[r.name] = ids
        # 同色批号分组仅供报告/人工核对，可行性判定绝不按颜色并库。
        self.colors_by_rgb: dict[RGB, tuple[str, ...]] = {}
        by_rgb: dict[RGB, list[str]] = {}
        for lot in lots:
            by_rgb.setdefault(lot.rgb, []).append(lot.lot_id)
        self.colors_by_rgb = {rgb: tuple(ids) for rgb, ids in by_rgb.items()}

    # -- 流量网络：源→批号(容量=罐数)→区(兼容边,容量1)→汇 ----------------- #
    def _build_network(
        self, forced: tuple[str, str] | None = None
    ) -> tuple[_Dinic, dict, int, int]:
        region_names = [r.name for r in self.regions]
        lot_ids = list(self.lots)
        source = 0
        lot_node = {lot_id: 1 + i for i, lot_id in enumerate(lot_ids)}
        region_node = {name: 1 + len(lot_ids) + i for i, name in enumerate(region_names)}
        sink = 1 + len(lot_ids) + len(region_names)
        dinic = _Dinic(sink + 1)
        for lot_id in lot_ids:
            dinic.add_edge(source, lot_node[lot_id], self.lots[lot_id].cans)
        for name in region_names:
            dinic.add_edge(region_node[name], sink, 1)
        for name in region_names:
            allowed = self.compatible_lots[name]
            if forced is not None and forced[0] == name:
                allowed = (forced[1],)
                if forced[1] not in self.compatible_lots[name]:
                    # 强制批号本就不兼容：造一条容量 0 的网络，保证无解。
                    dinic.add_edge(lot_node[forced[1]], region_node[name], 0)
                    continue
            for lot_id in allowed:
                dinic.add_edge(lot_node[lot_id], region_node[name], 1)
        return dinic, region_node, source, sink

    def _extract_scheme(
        self, dinic: _Dinic, region_node: dict, lot_ids: list[str], lot_node: dict
    ) -> Scheme:
        paints: list[RegionPaint] = []
        colors: dict[str, RGB] = {}
        for name, node in region_node.items():
            chosen: str | None = None
            for edge in dinic.graph[node]:
                # 反向边（指向批号）带流量即该区由该批号供应。
                if edge["to"] in lot_node.values() and edge["cap"] == 1:
                    lot_index = edge["to"] - 1
                    chosen = lot_ids[lot_index]
                    break
            if chosen is None:  # pragma: no cover - 仅在调用方误用满流结果时发生
                raise CabinetUnsatisfiable("流量未覆盖全部饰面区")
            lot = self.lots[chosen]
            paints.append(RegionPaint(region=name, lot_id=chosen, rgb=lot.rgb))
            colors[name] = lot.rgb
        return Scheme(colors=colors, paints=tuple(paints))

    def _solve(
        self, forced: tuple[str, str] | None = None
    ) -> Scheme | None:
        region_names = [r.name for r in self.regions]
        lot_ids = list(self.lots)
        lot_node = {lot_id: 1 + i for i, lot_id in enumerate(lot_ids)}
        dinic, region_node, source, sink = self._build_network(forced)
        flow = dinic.max_flow(source, sink)
        if flow != len(region_names):
            return None
        return self._extract_scheme(dinic, region_node, lot_ids, lot_node)

    def is_satisfiable(self) -> bool:
        return self._solve() is not None

    def any_scheme(self) -> Scheme:
        scheme = self._solve()
        if scheme is None:
            raise CabinetUnsatisfiable("漆罐库存/兼容关系无法配齐所有饰面区，柜约无解")
        return scheme

    def scheme_with_region_color(self, region_name: str, rgb: RGB) -> Scheme | None:
        """指定区刷指定 RGB：逐个尝试能刷出该色且兼容该区的批号，绝不并库。"""
        if region_name not in self._region_by_name:
            raise CabinetValidationError(f"未知饰面区 {region_name}")
        for lot_id in self.compatible_lots[region_name]:
            lot = self.lots[lot_id]
            if lot.rgb != rgb:
                continue
            scheme = self._solve(forced=(region_name, lot_id))
            if scheme is not None:
                return scheme
        return None

    def supports_scheme(self, scheme: Scheme) -> bool:
        """照单核对：逐区批号兼容且各批号领用量不超库存；不实际扣库存。"""
        if len(scheme.paints) != len(self.regions):
            return False
        used: dict[str, int] = {}
        for paint in scheme.paints:
            region = self._region_by_name.get(paint.region)
            if region is None:
                return False
            if paint.lot_id not in self.lots:
                return False
            if paint.lot_id not in self.compatible_lots[paint.region]:
                return False
            if self.lots[paint.lot_id].rgb != paint.rgb:
                return False
            used[paint.lot_id] = used.get(paint.lot_id, 0) + 1
        return all(used.get(lot_id, 0) <= lot.cans for lot_id, lot in self.lots.items())
