import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据归一化    Нормализовать данные
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])

# 读取 CIFAR-10 数据集    Чтение данных
train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_data = train_set.data / 255.0
mean = [round(train_data[:, :, :, 0].mean(), 4), round(train_data[:, :, :, 1].mean(), 4), round(train_data[:, :, :, 2].mean(), 4)]
std = [round(train_data[:, :, :, 0].std(), 4), round(train_data[:, :, :, 1].std(), 4), round(train_data[:, :, :, 2].std(), 4)]

transform_train.transforms.append(transforms.Normalize(mean, std))
transform_test.transforms.append(transforms.Normalize(mean, std))

# 打乱数据并划分数据集与验证集    Различайте обучающие и тестовые наборы
num_train = len(train_set)
indices = list(range(num_train))
split = int(np.floor(0.1 * num_train))

# 随机打乱数据    Рандомизировать данные
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

train_loader = DataLoader(train_set, batch_size=80, sampler=train_sampler)
valid_loader = DataLoader(train_set, batch_size=80, sampler=valid_sampler)
test_loader = DataLoader(test_set, batch_size=80, shuffle=True)


# 定义 Net 模型架构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> MaxPool -> Dropout
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2)
        )

        # Second block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> MaxPool -> Dropout
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3)
        )

        # Third block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> MaxPool -> Dropout
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.4)
        )

        # Fourth block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> MaxPool -> Dropout
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.5)
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(256 * 2 * 2, 10)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.flatten(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)
print(summary(model, input_size=(3, 32, 32)))

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


n_epochs = 300
train_loss_list = []
valid_loss_list = []
test_loss_list = []
train_acc_list = []
valid_acc_list = []
test_acc_list = []

for epoch in range(n_epochs):
    print('EPOCH {}:'.format(epoch + 1))

    def training(train_loader):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = y_pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = 100. * correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        return train_loss, train_accuracy


    def validation(valid_loader):
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                y_pred = model(inputs)
                loss = criterion(y_pred, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = y_pred.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        valid_loss = val_loss / len(valid_loader.dataset)
        valid_accuracy = 100. * correct / total
        print("Validation accuracy %.3f%%" % (valid_accuracy))
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_accuracy)
        return valid_loss, valid_accuracy


    def test(test_loader):
        model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * test_correct / total
        print("Test accuracy %.3f%%" % (test_accuracy))
        test_loss_list.append(test_loss)
        test_acc_list.append(test_accuracy)
        return test_loss, test_accuracy


    train_loss, train_acc = training(train_loader)
    val_loss, val_acc = validation(valid_loader)
    test_loss, test_acc = test(test_loader)

print("Complete!")




# 绘制学习曲线
plt.figure(figsize=(15, 6))

# 绘制训练与验证损失
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss', color='#8502d1')
plt.plot(valid_loss_list, label='Validation Loss', color='darkorange')
plt.plot(test_loss_list, label='Test Loss', color='green')
plt.legend()
plt.title('Loss Evolution')

# 绘制训练与验证准确率
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Accuracy', color='#8502d1')
plt.plot(valid_acc_list, label='Validation Accuracy', color='darkorange')
plt.plot(test_acc_list, label='Test Accuracy', color='green')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()
