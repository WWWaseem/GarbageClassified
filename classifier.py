import torch
import torch.nn as nn
import resneXt


# 经典工具管理变量
class AverageMeter:
    def __init__(self):
        self.val = 0  # 值
        self.avg = 0  # 平均值
        self.sum = 0  # 和
        self.count = 0  # 数量

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets):
    """ 准确率统计
        topk函数返回值: 第一列结果, 第二列索引
        取出二列结果再转置为行矩阵[[a, b, c]]
        取出一维矩阵[a, b, c] -> 数字化 并统计个数
        """
    pred = outputs.topk(k=1, dim=1, largest=True, sorted=True)[1].t()[0]
    predBool = pred.eq(targets.expand_as(pred))
    predVal = predBool.int().sum(0)
    return predVal.mul(100.0 / len(targets))


class GarbageClassifier:

    def __init__(self, num_classes):
        # 使用一个在大的数据集上预训练好的模型在自己数据上微调往往可以得到比直接用自己数据训练更好的效果
        # 预训练的模型参数从微调一开始就处于一个较好的位置，这样微调能够更快的使网络收敛
        model = resneXt.resnext101_32x16d_wsl()  # 加载模型

        # 停止部分参数训练 → 内存不够难以训练
        # 因为把所有权重都设置成受梯度影响训练速度太慢
        # 所以打算就把这个做一个修改经典网络来分类自己数据集的实验
        for param in model.parameters():
            param.requires_grad = False

        # 重新调整最后一层全连接层输出
        # 多层小卷积核堆叠相较于大卷积核可以引入更多的非线性
        input_feat = model.fc.in_features  # 获取模型中全连接层的输入特征
        model.fc = nn.Sequential(
            nn.Linear(in_features=input_feat, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=num_classes),
            nn.LogSoftmax(dim=1)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)
        self.model = model

    def train_model(self, train_loader, criterion, optimizer):
        # 定义变量
        losses = AverageMeter()
        top1 = AverageMeter()
        self.model.train()

        for batch_index, (inputs, targets) in enumerate(train_loader):
            # 模型结果预测
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)  # 获取输出
            loss = criterion(outputs, targets)  # 计算损失

            # 模型参数更新
            # grad在反向传播的过程中是累加的，也就是说上一次反向传播的结果会对下一次的反向传播的结果造成影响
            # 意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需要把梯度清零
            optimizer.zero_grad()  # 反向传播之前梯度归零
            loss.backward()  # 反向传播得到每个参数的梯度值
            optimizer.step()  # 梯度下降参数更新

            # 变量更新
            pre = accuracy(outputs.data, targets.data)
            losses.update(loss.item(), len(inputs))
            top1.update(pre.item(), len(inputs))
        return losses.avg, top1.avg

    def test_model(self, val_loader, criterion, test=None):
        # 定义变量
        losses = AverageMeter()
        top1 = AverageMeter()
        # 测试时, 应该用整个训练好的模型, 因此不需要dropout, 这就导致预测值和训练值的大小是不一样的
        # 测试时, 单个训练数据不必要去计算的均值和方差, 没必要标准化
        self.model.eval()

        for batch_index, (inputs, targets) in enumerate(val_loader):
            # 模型结果预测
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)  # 获取输出
            loss = criterion(outputs, targets)  # 计算loss

            # 变量更新
            pre = accuracy(outputs.data, targets.data)  # 返回值必须是两个，否则返回一个值的时候是个list，不方便处理第二个元素的大小
            losses.update(loss.item(), len(inputs))
            top1.update(pre.item(), len(inputs))

            return losses.avg, top1.avg
