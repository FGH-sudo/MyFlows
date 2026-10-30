from contextlib import nullcontext

from ..core.device import xp
from ..core.node import Variable
from ..core.graph import Graph
from ..core.tensor import Tensor
from ..ops.fused import adam_update

class Optimizer:
    """
    优化器基类
    用于实现各种优化算法（如SGD、Adam等）
    """
    def __init__(self, graph, learning_rate):
        """
        初始化优化器
        """
        self.graph = graph
        # self.target = target
        self.learning_rate = learning_rate
        self.acc_gradient = {}  # 累加梯度的字典
        self.acc_no = 0  # 累加次数
    
    def one_step(self, timer=None):
        """
        执行一步训练，包括前向传播和反向传播，并增加累加次数。
        timer: 可选，timer("forward") / timer("backward") 返回上下文管理器，分别包住前向与反向（含梯度累加）
        """
        # 前向传播和反向传播
        self.forward_backward(timer)
        # 增加累加次数
        self.acc_no += 1
    
    def get_gradient(self, node):
        """
        根据节点获取累加的平均梯度（累加梯度除以累加次数）
        """
        if node not in self.acc_gradient:
            return None

        if self.acc_no in (0, 1):
            # 只累加了一次时平均梯度就是累加值本身，省去一次整张量除法与分配
            return self.acc_gradient[node]

        return self.acc_gradient[node] / self.acc_no
        
    def _update(self, avg_gradients):
        """
        抽象方法，由子类实现具体参数更新逻辑。
        """
        raise NotImplementedError("_update() 必须由子类实现")
         
    def apply_gradients(self, gradients, summarize=False):
        """
        将外部计算的梯度应用到优化器中
        summarize: 是否累加梯度，True表示累加，False表示直接设置
        """
        
        if summarize:                                 # 适用于 mini-batch 训练的梯度累积场景
            # 累加梯度
            for node, gradient in gradients.items():
                if gradient is None:
                    continue  # 梯度为空，跳过  
                gradient = xp.asarray(Tensor.ensure(gradient).data)
                if node in self.acc_gradient:
                    self.acc_gradient[node] += gradient
                else:
                    self.acc_gradient[node] = gradient
        else:
            # 直接设置梯度
            for node, gradient in gradients.items():
                self.acc_gradient[node] = None if gradient is None else xp.asarray(Tensor.ensure(gradient).data)

    def update(self, var_gradients=None):
        """
        基类 update：处理累积梯度，然后调用子类 _update 完成具体更新。
        """
        # 如果外部提供梯度，就应用到累积梯度
        if var_gradients is not None:
            self.apply_gradients(var_gradients, summarize=False)

        # 调用子类实现的具体更新逻辑
        self._update()

        # 清空累积梯度和次数
        self.acc_gradient.clear()
        self.acc_no = 0
        
    def forward_backward(self, timer=None):
        """
        执行前向传播和反向传播
        计算目标节点对各个可训练节点的梯度，并累加到acc_gradient中
        """
        # 前向传播
        with (timer("forward") if timer else nullcontext()):
            self.graph.forward()

        with (timer("backward") if timer else nullcontext()):
            # 反向传播
            self.graph.backward()
            self._accumulate_gradients()

    def _accumulate_gradients(self):
        # 累加梯度
        for node in self.graph.nodes:
            if node.trainable is True:
                if node.grad is not None:
                    if node in self.acc_gradient:
                        self.acc_gradient[node] += node.grad
                    else:
                        self.acc_gradient[node] = node.grad


class MBGD(Optimizer):
    def __init__(self, graph, learning_rate=0.01):
        super().__init__(graph, learning_rate)

    def _update(self):
        # 计算平均梯度
        avg_gradients = {}
        for node in self.acc_gradient:
            grad = self.get_gradient(node)
            if grad is not None:
                avg_gradients[node] = grad

        # 执行参数更新
        for node, grad in avg_gradients.items():
            if getattr(node, "trainable", False) and grad is not None:
                node.value -= self.learning_rate * grad


class Momentum(Optimizer):
    def __init__(self, graph, learning_rate=0.01, momentum=0.9):
        super().__init__(graph, learning_rate)
        self.momentum = momentum
        self.v = {}  # 储存每个可训练变量的动量

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:  # 如果节点是可训练的变量
                # 获取该变量在当前epoch的平均梯度
                grad = self.get_gradient(node)
                if grad is None:
                    continue
                # 若该节点不在动量字典里，将当前梯度作为初始动量
                if node not in self.v:
                    self.v[node] = -self.learning_rate * grad
                else:
                    # 更新动量
                    self.v[node] = self.momentum * self.v[node] - self.learning_rate * grad
                node.value += self.v[node]


class AdaGrad(Optimizer):
    def __init__(self, graph, learning_rate=0.01, eps=1e-8):
        super().__init__(graph, learning_rate)
        self.eps = eps
        self.s = {}  # 储存每个可训练变量的历史梯度平方

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # 获取变量平均梯度
                grad = self.get_gradient(node)
                if grad is None:
                    continue
                if node not in self.s:
                    self.s[node] = xp.power(grad, 2)
                else:
                    self.s[node] += xp.power(grad, 2)
                # 更新变量值
                node.value -= self.learning_rate * grad / (xp.sqrt(self.s[node] + self.eps))


class RMSProp(Optimizer):
    def __init__(self, graph, learning_rate=0.01, beta=0.9, eps=1e-8):
        super().__init__(graph, learning_rate)
        self.beta = beta
        self.eps = eps
        self.s = {}  # 储存每个可训练变量的历史梯度平方的加权累计(引入指数加权移动平均)

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # 获取变量平均梯度
                grad = self.get_gradient(node)
                if grad is None:
                    continue
                if node not in self.s:
                    self.s[node] = (1 - self.beta) * xp.power(grad, 2)
                else:
                    self.s[node] = self.beta * self.s[node] + (1 - self.beta) * xp.power(grad, 2)
                # 更新变量值
                node.value -= self.learning_rate * grad / (xp.sqrt(self.s[node] + self.eps))


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015)。

    一阶矩 v 与二阶矩 s 从 0 开始，m_t = b1 m + (1 - b1) g，s_t = b2 s + (1 - b2) g^2，
    再按 1 - b^t 修正偏差。旧实现首步直接令 v = g、s = g^2 后仍做偏差修正，首步步长只有
    标准值的约 0.32 倍，且 g_1^2 在 s 中被放大 1000 倍、上千步才衰减。
    每个参数一次融合更新（GPU 为单个 kernel），CPU 回退与 GPU 的逐步舍入顺序一致。
    """

    def __init__(self, graph, learning_rate=0.01, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__(graph, learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.v = {}  # 一阶矩：历史梯度的指数加权平均（Momentum）
        self.s = {}  # 二阶矩：历史梯度平方的指数加权平均（RMSProp）
        self.t = 0   # 记录时间步长，用以修正偏差

    def _update(self):
        # 每次更新，时间步长+1
        self.t += 1
        # 平均梯度 = 累加梯度 / 累加次数，作为缩放因子传入融合 kernel，不单独生成新数组
        scale = 1.0 if self.acc_no in (0, 1) else 1.0 / self.acc_no

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                grad = self.acc_gradient.get(node)
                if grad is None:
                    continue
                if node not in self.s:
                    self.v[node] = xp.zeros_like(node.value)
                    self.s[node] = xp.zeros_like(node.value)
                adam_update(node.value, grad, self.v[node], self.s[node], lr=self.learning_rate,
                            beta1=self.beta_1, beta2=self.beta_2, eps=self.eps, step=self.t,
                            grad_scale=scale)
