import torch
import torch.nn as nn


# 残差块
class BasicBlock(nn.Module):     # 基本残差模块

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # 收缩子模块
        self.shrinkage = Shrinkage(out_channels, gap_size=1)
        # 卷积、卷积、收缩
        self.residual_function = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),       # 计算结果直接覆盖tensor减少内存
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            self.shrinkage
        )
        # 连接点
        self.shortcut = nn.Sequential()

        # 通过1*1卷积来统一维度
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


# 收缩子模块
class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):       # out channel
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)  # 实现GAP 全局平均池化，每个通道数输出1
        self.fc = nn.Sequential(  # 学习阈值的两层FC
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),     # 输出压缩到0,1之间
        )

    def forward(self, x):
        x_raw = x  # 原始值 x
        x = torch.abs(x)  # 取绝对值
        x_abs = x
        x = self.gap(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展开
        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        x = self.fc(x)  # 通过两层FC
        x = torch.mul(average, x)  # 与绝对值点乘
        x = x.unsqueeze(2)  # 增加两个维度，获得  阈值向量
        # 软阈值化
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)  # 缩放系数
        x = torch.mul(torch.sign(x_raw), n_sub)  # 对原始值x进行缩放(sign返回符号1 0 -1)
        return x


# 整个网络结构
class RSNet(nn.Module):

    def __init__(self, block, num_classes):   # 此处block参数是上面的BasicBlock类
        super().__init__()
        self.in_channels = 4

        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1, stride=2, bias=False))

        # 堆叠模块
        self.conv2_x = self._make_layer(block, 4, 2)
        self.conv3_x = self._make_layer(block, 4, 1)
        self.conv4_x = self._make_layer(block, 8, 2)
        self.conv5_x = self._make_layer(block, 8, 1)
        self.conv6_x = self._make_layer(block, 16, 2)
        self.conv7_x = self._make_layer(block, 16, 1)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def _make_layer(self, block, out_channels, stride):
        # 用模块构建层
        layers = block(self.in_channels, out_channels, stride)
        self.in_channels = out_channels            # 改变channel与下一层计算

        return nn.Sequential(layers)

    def forward(self, x):  # batch_size * 1 * 2560
        output = self.conv1(x)  # batch_size * 4 * 1280
        output = self.conv2_x(output)  # batch_size * 4 * 640
        output = self.conv3_x(output)  # batch_size * 4 * 640
        output = self.conv4_x(output)  # batch_size * 8 * 320
        output = self.conv5_x(output)  # batch_size * 8 * 320
        output = self.conv6_x(output)  # batch_size * 16 * 160
        output = self.conv7_x(output)  # batch_size * 16 * 160
        output = self.bn(output)
        output = self.relu(output)
        output = self.avg_pool(output)  # batch_size * 16 * 1
        output = output.view(output.size(0), -1)  # batch_size * 16
        output = self.fc(output)  # batch_size * num_classes

        return output


def Net(num_classes):        # 类别数
    """ return a RSNet 18 object
    """
    return RSNet(BasicBlock, num_classes)
