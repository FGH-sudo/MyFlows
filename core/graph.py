from .device import xp
from .node import Node, Variable
from .graph_opt import apply_graph_optimizations

class Graph:
    _active_plan = None
    
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
        self.profiler = None
        self._leaves = [node for node in self.nodes if not node.parents]
        self._retained = set()
        self._signature = None
        self._forward_active = None
        self._refresh_gradients()

    def retain_grad(self, *nodes):
        """Request intermediate adjoints, e.g. an otherwise frozen Grad-CAM feature."""
        for node in nodes:
            if node not in self.nodes or (not node.parents and not node.requires_grad):
                raise ValueError("retain_grad expects a graph intermediate or a differentiable leaf")
        self._retained.update(node for node in nodes if node.parents)
        self._signature = None

    def _refresh_gradients(self):
        """Cache a topological plan; changing leaf flags or retained targets invalidates it."""
        signature = tuple(node.requires_grad for node in self._leaves)
        if signature != self._signature:
            needed = dict(zip(self._leaves, signature))
            propagate = []
            for node in self.nodes:
                if node.parents:
                    parents_need_grad = any(needed[p] for p in node.parents)
                    needed[node] = parents_need_grad or node in self._retained
                    if parents_need_grad:
                        propagate.append(node)
            self._active = frozenset(node for node, required in needed.items() if required)
            self._flags = tuple(needed.items())
            self._backward_nodes = tuple(reversed(propagate))
            self._clear_nodes = tuple(node for node in self.nodes if node in self._active)
            self._signature = signature
            self._plan = object()
        if Graph._active_plan is not self._plan:
            # A second graph can share these nodes but request another intermediate gradient.
            for node, required in self._flags:
                if node.parents:
                    node.requires_grad = required
                if not required:
                    node.grad = None
            Graph._active_plan = self._plan
    
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
        self._refresh_gradients()
        self._forward_active = self._active
        for node in self.sorted_nodes:
            if node.parents:
                inputs = [p.value for p in node.parents]
            elif isinstance(node, Variable):
                inputs = ()
            else:
                continue
            if self.profiler is None:
                node.forward(*inputs)
            else:
                with self.profiler.node(node, "forward"):
                    node.forward(*inputs)

    def backward(self):
        self._refresh_gradients()
        if self._forward_active is not None and self._forward_active is not self._active:
            if not self._active.issubset(self._forward_active):
                raise RuntimeError("new gradients requested after forward; run forward again to rebuild saved tensors")
        if self.target_node not in self._active:
            return
        for n in self._clear_nodes:
            n.clear_grad()
            
        self.target_node.grad = xp.ones_like(self.target_node.value)
        
        # 使用预先计算好的反向排序列表
        for node in self._backward_nodes:
            if self.profiler is None:
                node.backward()
            else:
                with self.profiler.node(node, "backward"):
                    node.backward()
