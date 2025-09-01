import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConv(nn.Module):
    def __init__(self, input_dim=1536, out_dim=512):
        super(MultiScaleConv, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, out_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, out_dim, kernel_size=7, padding=3)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)  
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        
        x = torch.cat((x1, x2, x3), dim=1)
        
        return x

class VisionDangerClassification(nn.Module):
    def __init__(self, input_dim=1536, num_classes=3, heads=8, dropout_rate=0.5):
        super(VisionDangerClassification, self).__init__()
        
        self.multi_scale_conv = MultiScaleConv(input_dim=input_dim, out_dim=512)
        
        self.attn = nn.MultiheadAttention(embed_dim=512*3, num_heads=heads, dropout=dropout_rate)
        
        self.fc1 = nn.Linear(512*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)  # 3个类别
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        multi_scale_features = self.multi_scale_conv(x)
        multi_scale_features = multi_scale_features.permute(2,0,1)  
        attn_output, _ = self.attn(multi_scale_features, multi_scale_features, multi_scale_features)
        x = torch.relu(self.fc1(attn_output.mean(dim=0)))  
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  
        
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float()
        p_t = (inputs * targets).sum(dim=1)
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, outputs, targets):
        n_class = outputs.size(1)
        one_hot = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        loss = -(one_hot * F.log_softmax(outputs, dim=1)).sum(dim=1)
        return loss.mean()


def total_loss(logits, labels, focal_loss_fn=None, label_smoothing_fn=None):
    cross_entropy_loss = F.cross_entropy(logits, labels)  
    focal_loss = focal_loss_fn(logits, labels) if focal_loss_fn else 0
    label_smoothing_loss = label_smoothing_fn(logits, labels) if label_smoothing_fn else 0
    
    return cross_entropy_loss + focal_loss + label_smoothing_loss  

model = VisionDangerClassification(input_dim=1536, num_classes=3)

inputs = torch.randn(32, 646, 1536)  
labels = torch.randint(0, 3, (32,))  

logits = model(inputs)

features = logits  
focal_loss_fn = FocalLoss()
label_smoothing_fn = LabelSmoothingLoss(smoothing=0.1)

loss = total_loss(logits, labels, focal_loss_fn=focal_loss_fn, label_smoothing_fn=label_smoothing_fn)
print(f"Total Loss: {loss.item()}")
