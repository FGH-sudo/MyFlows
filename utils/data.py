import numpy as np
import matplotlib.pyplot as plt
from ..core.graph import Graph


# 随机数据生成
class DataGenerator:
    """
    模拟数据生成器
    每个人包含：
        - 身高（cm）
        - 体重（kg）
        - 体脂率（%）
    规则：
        如果 height + weight - 100 * body_fat >= 200 → label = 1（男）
        否则 label = -1（女）
    """
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.noise_level = 0.0  # 噪声强度

    def set_noise(self, level=0.05):
        """设置噪声比例"""
        self.noise_level = level

    def generate(self, num_samples=100):
        """生成模拟数据"""
        height = np.random.normal(168, 8, num_samples)
        weight = np.random.normal(65, 10, num_samples)
        body_fat = np.random.normal(0.20, 0.05, num_samples)

        if self.noise_level > 0:
            height += np.random.normal(0, self.noise_level * np.std(height), num_samples)
            weight += np.random.normal(0, self.noise_level * np.std(weight), num_samples)
            body_fat += np.random.normal(0, self.noise_level * np.std(body_fat), num_samples)

        score = height + weight - 100 * body_fat
        label = np.where(score >= 200, 1, -1)

        X_raw = np.stack([height, weight, body_fat], axis=1)
    
        # --- 核心改动：在最左侧插入一列 1 ---
        # np.ones((行数, 1))
        bias_column = np.ones((num_samples, 1))
        X_data = np.hstack([bias_column, X_raw]) # 变成 (N, 4)
        
        X = np.array(X_data)
        y = np.array(label).reshape(-1, 1)
        
        return X, y


# 数据预处理模块
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        # 记录训练集的统计特性
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def standardize_data(self, X, epsilon=1e-5):
        """
        数据标准化
        公式: X_std = (X - mean) / std
        """
        # 计算每个特征（每个像素点）在所有样本上的均值和标准差
        # axis=0 表示对样本维度取平均
        return (X - self.mean) / (self.std + epsilon)


def min_max_scale(data):
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    return (data - col_min) / (col_max - col_min)


def augment_data(X, noise_level=0.01):
    """
    数据增强：添加微量噪声
    保持数字中心对齐，仅改变像素清晰度
    """
    # 产生与 X 形状相同的高斯噪声
    noise = np.random.normal(0, noise_level, X.shape)
    
    # 将噪声叠加到图像上
    X_aug = X + noise
    
    # 裁剪边缘，确保像素值依然在合理范围内 (假设已标准化)
    # 如果是 0-1 归一化，使用 np.clip(X_aug, 0, 1)
    return X_aug


# 梯度数值检验 
def gradient_check(graph, loss_node, target_layer, epsilon=1e-7):
    """
    通过数值模拟验证解析梯度的准确性
    """
    print("\n" + "="*20 + " 梯度数值检验 " + "="*20)
    
    # 彻底清空现有梯度 
    for node in graph.nodes:
        if hasattr(node, 'grad') and node.grad is not None:
            node.grad = np.zeros_like(node.grad)

    # 计算解析梯度 
    graph.forward()
    graph.backward() # 显式触发反向传播
    analytical_grad = target_layer.W.grad[0, 0] 

    # 计算数值梯度 
    original_value = target_layer.W.value[0, 0]
    
    target_layer.W.value[0, 0] = original_value + epsilon
    graph.forward()
    loss_plus = loss_node.value
    
    target_layer.W.value[0, 0] = original_value - epsilon
    graph.forward()
    loss_minus = loss_node.value
    
    # 恢复原值
    target_layer.W.value[0, 0] = original_value
    
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    
    # 对比误差
    diff = np.abs(analytical_grad - numerical_grad)
    print(f"节点 [{target_layer.name}] 的权重 W[0,0]:")
    print(f"  - 解析梯度 (MyFlows): {analytical_grad:.8f}")
    print(f"  - 数值梯度 (Simulation): {numerical_grad:.8f}")
    print(f"  - 绝对误差: {diff:.2e}")
    
    if diff < 1e-6:
        print("校验通过！")
    else:
        print("校验失败。")


# 计算图可视化
def plot_computational_graph(graph):
    print("\n" + " =" * 10 + " 数据流向图 " + " =" * 10)
    
    flow_path = []
    for node in graph.nodes:
        # 获取节点显示名称
        name = getattr(node, 'name', node.__class__.__name__)
        
        # 记录形状
        shape = ""
        if hasattr(node, 'value') and node.value is not None:
            if hasattr(node.value, 'shape'):
                shape = f"({node.value.shape})"
        
        flow_path.append(f"[{name}{shape}]")

    # 用箭头连接并打印
    print("\n  " + "  -->  \n  ".join(flow_path))
    print("\n" + " =" * 30)

    