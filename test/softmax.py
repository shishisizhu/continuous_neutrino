import torch
import torch.nn as nn
import time

# 设置 GPU (如果可用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
batch_size = 64
seq_len = 128
num_features = 128
iterations = 1000000 # 循环次数

# 创建样本数据，对 Softmax 来说，常见的是 `[batch_size, seq_len, num_features]` 形状
data_softmax = torch.randn(batch_size, seq_len, num_features, device=device)

# 初始化 Softmax
softmax_dim_2 = nn.Softmax(dim=2).to(device)  # 对最后一个维度 (features) 进行 softmax
softmax_dim_1 = nn.Softmax(dim=1).to(device)  # 对中间维度 (seq_len) 进行 softmax

# CUDA 同步函数
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# 测试 Softmax (dim=2) 的总耗时
print("Testing Softmax (dim=2):")
cuda_sync()  # 同步以确保之前的操作完成
start_softmax_dim_2 = time.time()
for _ in range(iterations):
    _ = softmax_dim_2(data_softmax)  # 执行前向传播
cuda_sync()  # 确保 GPU 任务全部完成
end_softmax_dim_2 = time.time()
print(f"Softmax (dim=2) {iterations} iterations total time: {end_softmax_dim_2 - start_softmax_dim_2:.6f} seconds")
''' 
# 测试 Softmax (dim=1) 的总耗时
print("\nTesting Softmax (dim=1):")
cuda_sync()  # 同步以确保之前的操作完成
start_softmax_dim_1 = time.time()
for _ in range(iterations):
    _ = softmax_dim_1(data_softmax)  # 执行前向传播
cuda_sync()  # 确保 GPU 任务全部完成
end_softmax_dim_1 = time.time()
print(f"Softmax (dim=1) {iterations} iterations total time: {end_softmax_dim_1 - start_softmax_dim_1:.6f} seconds")
'''
