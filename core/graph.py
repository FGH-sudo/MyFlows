import numpy as np
from .node import Node, Variable
from .graph_opt import apply_graph_optimizations

class Graph:
    
    def __init__(
        self,
        target_node,
        optimize=False,
        fold_constants=False,
        fuse_linear=True,
        fuse_activations=True,
    ):
        """
        :param optimize: 为 True 时在构图后运行可选构建期优化（如常量折叠）。
        :param fold_constants: 仅在 ``optimize=True`` 时生效，是否折叠常量 Add。
        """
        self._opt_stats = None
        if optimize:
            self.target_node, self._opt_stats = apply_graph_optimizations(
                target_node,
                fold_constants=fold_constants,
                fuse_linear=fuse_linear,
                fuse_activations=fuse_activations,
            )
        else:
            self.target_node = target_node

        # 自动构建拓扑排序（优化后再排序，避免包含已被替换的节点）
        self.sorted_nodes = self._build_topo_sort(self.target_node)
        
        # self.nodes 是所有节点的集合
        
        self.nodes = self.sorted_nodes
    
    def _build_topo_sort(self, target_node):
        
        
        visited, order = set(), []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for p in getattr(node, "parents", []):
                dfs(p)
            order.append(node)
        
        
        dfs(target_node)
        return order

    def forward(self):
        
        for node in self.sorted_nodes:
            if node.parents:
                inputs = [p.value for p in node.parents]
                node.forward(*inputs)
            elif isinstance(node, Variable):
                # Variable 节点是起点
                node.forward() 

    def backward(self):
        
        # 清空梯度
        for n in self.nodes:
            n.clear_grad()
            
        self.target_node.grad = 1.0 # 目标节点梯度为1
        
        # 使用预先计算好的反向排序列表
        for node in reversed(self.sorted_nodes):
            node.backward()
