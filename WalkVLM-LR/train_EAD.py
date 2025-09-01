import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from EAD import VisionDangerClassification

class VideoDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.video_folders = self._get_video_folders(data_dir)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["A", "B", "C"])  
        self.visual_features = []
        self.labels = []
        self._load_data()

    def _get_video_folders(self, data_dir):
        video_folders = []
        for root, dirs, files in os.walk(data_dir):
            for dir_name in dirs:
                video_folders.append(os.path.join(root, dir_name))
        print(f"Found {len(video_folders)} video folders.") 
        return video_folders

    def _load_data(self):
        for video_folder in self.video_folders:
            print(f"Processing video folder: {video_folder}")  

            json_file_path = os.path.join(video_folder, f"{os.path.basename(video_folder)}_features.json")
            if not os.path.exists(json_file_path):
                print(f"Warning: JSON file not found for {video_folder}. Skipping.")
                continue

            with open(json_file_path, 'r') as f:
                frames_data = json.load(f)
            
            print(f"Processing {len(frames_data)} frames in {json_file_path}.") 

            for frame_data in frames_data:
                visual_features = np.array(frame_data['visual_features'], dtype=np.float32)

                label = frame_data['label'].split()[1] 
                
                try:
                    label = self.label_encoder.transform([label])[0]
                except ValueError:
                    print(f"Unknown label found: {label}. Skipping this entry.")
                    continue  
                
                self.visual_features.append(torch.tensor(visual_features, dtype=torch.float32))
                self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.visual_features)

    def __getitem__(self, idx):
        return self.visual_features[idx], self.labels[idx]

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=3):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class VisualDangerClassificationHead(nn.Module):
    def __init__(self, input_dim=1536, num_classes=3, dropout_rate=0.5):
        super(VisualDangerClassificationHead, self).__init__()
        
        self.pooling = nn.AdaptiveAvgPool1d(1) 
        
        self.fc1 = nn.Linear(input_dim, 1024)   
        self.fc2 = nn.Linear(1024, 512)         
        self.fc21 = nn.Linear(512, 512)
        self.fc22 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)  
        
        self.dropout = nn.Dropout(dropout_rate) 

    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.pooling(x)    
        x = x.squeeze(-1)       
        x = torch.relu(self.fc1(x))   
        x = self.dropout(x)           
        x = torch.relu(self.fc2(x))   
        x = torch.relu(self.fc21(x))   
        x = torch.relu(self.fc22(x))   
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


def init_distributed_training(backend='nccl'):
    dist.init_process_group(backend)
    torch.cuda.set_device(dist.get_rank())

def get_data_loader(data_dir, batch_size=32, num_workers=8):
    dataset = VideoDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def train_model(model, train_loader, epochs=60, batch_size=32, lr=1e-4):
    model.train()
    focal_loss_fn = FocalLoss()
    label_smoothing_fn = LabelSmoothingLoss(smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = total_loss(outputs, targets, focal_loss_fn=focal_loss_fn, label_smoothing_fn=label_smoothing_fn)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)

        epoch_loss = running_loss / len(train_loader)
        accuracy = (correct_preds / total_preds) * 100

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

def main():
    init_distributed_training()

    data_dir = ""  
    batch_size = 16
    num_workers = 8
    train_loader = get_data_loader(data_dir, batch_size, num_workers)

    # 初始化模型
    input_dim = 364 * 1536  
    model = VisionDangerClassification(input_dim=1536, num_classes=3)
    model = model.cuda()

    model = DDP(model, device_ids=[torch.cuda.current_device()])

    train_model(model, train_loader)

if __name__ == "__main__":
    main()
