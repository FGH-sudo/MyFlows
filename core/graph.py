from .device import xp
from .node import GradientOp, Node, Variable
from .graph_opt import apply_graph_optimizations, batchnorm_activation_groups

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
        self._bn_groups = batchnorm_activation_groups(self.nodes) if optimize and fuse_activations else {}
        if self._opt_stats is not None:
            self._opt_stats["bn_activation_fusions"] = len(self._bn_groups)
        self.profiler = None
        self._leaves = [node for node in self.nodes if not node.parents]
        self._retained = set()
        self._signature = None
        self._forward_active = None
        self._forward_groups = None
        self._materialize_values = False
        self._refresh_gradients()

    def materialize_intermediates(self, enabled=True):
        """Keep logical BN and residual values on the next forward for full graph inspection."""
        if self._materialize_values != bool(enabled):
            self._materialize_values = bool(enabled)
            self._signature = None

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
            groups = {} if self._materialize_values else {
                n: g for n, g in self._bn_groups.items() if self._retained.isdisjoint(g.hidden)}
            hidden = {node for group in groups.values() for node in group.hidden}
            self._group_signature = frozenset(groups)
            self._execution_nodes = tuple(groups.get(n, n) for n in self.nodes if n not in hidden)
            self._backward_nodes = tuple((groups.get(n, n), n) for n in reversed(propagate) if n not in hidden)
            self._legacy_parents = {n: tuple(p for p in n.parents if needed[p]) for n in propagate
                                    if not isinstance(n, GradientOp)}
            self._keep_grads = tuple(n for n in self.nodes if needed[n] and (not n.parents or n in self._retained))
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
        self._forward_groups = self._group_signature
        for node in self._execution_nodes:
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

    def backward(self, loss_scale=1.0):
        self._refresh_gradients()
        if self._forward_active is not None and self._forward_active is not self._active:
            if not self._active.issubset(self._forward_active) or self._forward_groups != self._group_signature:
                raise RuntimeError("new gradients requested after forward; run forward again to rebuild saved tensors")
        if self.target_node not in self._active:
            return
        for n in self._clear_nodes:
            n.grad = None
            
        self.target_node.grad = (xp.ones_like(self.target_node.value) if loss_scale == 1.0
                                 else xp.full_like(self.target_node.value, loss_scale))
        
        # 使用预先计算好的反向排序列表
        for node, output in self._backward_nodes:
            if output.grad is None:
                continue
            for parent in self._legacy_parents.get(node, ()):
                # Third-party Node subclasses may still write parent.grad += contribution.
                if parent.grad is None:
                    parent.clear_grad()
            saved = output.grad_tensor if output in self._retained else None
            if self.profiler is None:
                node.backward()
            else:
                with self.profiler.node(node, "backward"):
                    node.backward()
            output.grad_tensor = saved
        for node in self._keep_grads:
            if node.grad is None:
                node.clear_grad()
