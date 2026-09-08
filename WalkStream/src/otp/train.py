import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from otp_model import FocalLoss, LabelSmoothingLoss, VisionDangerClassification, total_loss


LABEL_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


class VideoDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir).expanduser().resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.samples = []
        for json_file in sorted(self.data_dir.rglob("*_features.json")):
            self.samples.extend(self._load_json_file(json_file))

        if not self.samples:
            raise ValueError(f"No '*_features.json' files were found under {self.data_dir}")

    @staticmethod
    def _parse_label(raw_label):
        if raw_label is None:
            return None
        label_text = str(raw_label).strip().split()[-1]
        return LABEL_TO_ID.get(label_text)

    def _load_json_file(self, json_file: Path):
        with open(json_file, "r", encoding="utf-8") as f:
            frames_data = json.load(f)

        loaded = []
        for frame_data in frames_data:
            visual_features = frame_data.get("visual_features")
            label_id = self._parse_label(frame_data.get("label"))
            if visual_features is None or label_id is None:
                continue

            loaded.append(
                (
                    torch.tensor(np.array(visual_features, dtype=np.float32), dtype=torch.float32),
                    torch.tensor(label_id, dtype=torch.long),
                )
            )
        return loaded

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    features, labels = zip(*batch)
    seq_lengths = [feature.size(0) for feature in features]
    max_len = max(seq_lengths)
    hidden_size = features[0].size(1)

    padded = torch.zeros(len(features), max_len, hidden_size, dtype=torch.float32)
    for idx, feature in enumerate(features):
        padded[idx, : feature.size(0)] = feature

    return padded, torch.stack(labels)


def train_model(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = VideoDataset(args.data_dir)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = VisionDangerClassification(input_dim=args.input_dim, num_classes=3).to(device)
    focal_loss_fn = FocalLoss()
    label_smoothing_fn = LabelSmoothingLoss(smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = total_loss(outputs, targets, focal_loss_fn=focal_loss_fn, label_smoothing_fn=label_smoothing_fn)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)

        epoch_loss = running_loss / max(len(train_loader), 1)
        accuracy = 100.0 * correct_preds / max(total_preds, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_id": LABEL_TO_ID,
            "config": {
                "input_dim": args.input_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "label_smoothing": args.label_smoothing,
            },
        },
        output_path,
    )
    print(f"Saved OTP classifier checkpoint to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the OTP environment awareness discriminator.")
    parser.add_argument("--data_dir", required=True, help="Directory containing '*_features.json' files.")
    parser.add_argument("--output_path", default="model.pth", help="Path to save the trained checkpoint.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_dim", type=int, default=1536)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. 'cuda:0' or 'cpu'.")
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
