import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# 超参数设置
# -------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

INPUT_SIZE = 784      # 例如：MNIST 图像展平 (28x28)
HIDDEN_SIZE = 256
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 60000  # 模拟 MNIST 大小

# -------------------------------
# 构造一个简单的模型（全连接网络）
# -------------------------------
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# -------------------------------
# 生成随机数据集（模拟真实数据）
# -------------------------------
torch.manual_seed(42)
X = torch.randn(DATASET_SIZE, INPUT_SIZE)
y = torch.randint(0, NUM_CLASSES, (DATASET_SIZE,))

# 创建 DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# 初始化模型、损失函数、优化器
# -------------------------------
model = SimpleNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------
# 训练函数
# -------------------------------
def train():
    model.train()
    total_loss = 0.0
    for data, target in dataloader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# -------------------------------
# 主训练循环 + 计时
# -------------------------------
print("Starting training...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    avg_loss = train()
    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Time: {epoch_time:.4f}s")

total_time = time.time() - start_time
print(f"Total training time: {total_time:.4f} seconds")

