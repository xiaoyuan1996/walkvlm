import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义多尺度卷积模块
class MultiScaleConv(nn.Module):
    def __init__(self, input_dim=1536, out_dim=512):
        super(MultiScaleConv, self).__init__()
        
        # 不同大小的卷积核提取多尺度特征
        self.conv1 = nn.Conv1d(input_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, out_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, out_dim, kernel_size=7, padding=3)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x形状 [batch_size, num_slices, embed_size] --> [batch_size, embed_size, num_slices]
        x = x.transpose(1, 2)  # 转置为 [batch_size, embed_size, num_slices]
        
        # 通过不同尺度的卷积进行特征提取
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        
        # 将不同尺度的特征拼接
        x = torch.cat((x1, x2, x3), dim=1)
        
        return x

# 定义ViT + FPN + 多尺度卷积的网络结构
class VisionDangerClassification(nn.Module):
    def __init__(self, input_dim=1536, num_classes=3, heads=8, dropout_rate=0.5):
        super(VisionDangerClassification, self).__init__()
        
        # 特征提取模块
        self.multi_scale_conv = MultiScaleConv(input_dim=input_dim, out_dim=512)
        
        # 使用官方的 MultiheadAttention 模块
        self.attn = nn.MultiheadAttention(embed_dim=512*3, num_heads=heads, dropout=dropout_rate)
        
        # 分类头
        self.fc1 = nn.Linear(512*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)  # 3个类别
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x的形状是 [batch_size, num_slices, embed_size]
        
        # 多尺度卷积模块提取特征
        multi_scale_features = self.multi_scale_conv(x)
        # print(multi_scale_features.shape)#[32, 1536, 646]
        # multi_scale_features = x.permute(0,2,1)
        
        # 通过官方的 MultiheadAttention 进行自注意力处理
        # 注意：MultiheadAttention expects input shape: (seq_len, batch_size, embed_dim)
        multi_scale_features = multi_scale_features.permute(2,0,1)  # 转置为 [num_slices, batch_size, embed_size]
        attn_output, _ = self.attn(multi_scale_features, multi_scale_features, multi_scale_features)
        # attn_output = x.permute(1,0,2)
        # 通过全连接层进行分类
        x = torch.relu(self.fc1(attn_output.mean(dim=0)))  # 对整个序列的输出做平均池化
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 最终输出 [batch_size, num_classes]
        
        return x

# 对比学习损失
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # 计算 softmax 输出
        inputs = F.softmax(inputs, dim=1)
        
        # 计算目标类别的概率
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # 计算焦点损失
        p_t = (inputs * targets).sum(dim=1)
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Label Smoothing Loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, outputs, targets):
        # 使用label smoothing处理标签
        n_class = outputs.size(1)
        one_hot = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        loss = -(one_hot * F.log_softmax(outputs, dim=1)).sum(dim=1)
        return loss.mean()

# 修改损失函数：将交叉熵损失与其他损失结合
def total_loss(logits, labels, focal_loss_fn=None, label_smoothing_fn=None):
    cross_entropy_loss = F.cross_entropy(logits, labels)  # 标准交叉熵损失
    
    # 选择性添加其他损失
    focal_loss = focal_loss_fn(logits, labels) if focal_loss_fn else 0
    label_smoothing_loss = label_smoothing_fn(logits, labels) if label_smoothing_fn else 0
    
    return cross_entropy_loss + focal_loss + label_smoothing_loss  # 合并损失

# 测试模型构建
model = VisionDangerClassification(input_dim=1536, num_classes=3)

# 随机输入数据和标签
inputs = torch.randn(32, 646, 1536)  # 输入形状 [batch_size, num_slices, embed_size]
labels = torch.randint(0, 3, (32,))  # 标签 [batch_size]

# 进行前向传播
logits = model(inputs)

# 假设 features 是模型的最后一层特征
features = logits  # 在实际应用中，可以从ViT中提取feature
focal_loss_fn = FocalLoss()
label_smoothing_fn = LabelSmoothingLoss(smoothing=0.1)


# 计算总损失
loss = total_loss(logits, labels, focal_loss_fn=focal_loss_fn, label_smoothing_fn=label_smoothing_fn)
print(f"Total Loss: {loss.item()}")
