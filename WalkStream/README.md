# WalkStream: Data and Reward-Enhanced Low-Latency VLM for Blind Walking Assistance

[![Paper](https://img.shields.io/badge/Paper-International%20Journal%20of%20Computer%20Vision%20Submited-blue)](https://arxiv.org/abs/xxx)[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)[![Python](https://img.shields.io/badge/Python-3.8+-orange.svg)](https://www.python.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

WalkStream is a specialized Vision-Language Model (VLM) designed for real-time walking assistance for the Blind and Low Vision (BLV) community. It addresses critical limitations of existing methods by providing concise, low-latency, and context-aware walking reminders through three core innovations: a large-scale standardized dataset (WRD), a reward-enhanced inference framework, and a decoupled asynchronous architecture.

## 🌟 Key Features
- **Large-Scale Standardized Dataset**: 98k annotated video clips (478 hours) from 44 countries, covering diverse scenes (urban/suburban/rural, day/night, various weather conditions).
- **Low Redundancy Output**: GRPO-based inference framework with four human-preference reward functions (conciseness, fluency, keyword density, accuracy) to generate concise yet informative reminders.
- **Ultra-Low Latency**: Decoupled asynchronous architecture with long-short-term memory buffer, reducing end-to-end latency by 52.6% compared to serial pipelines.
- **High Temporal Adaptability**: Online Triggered Prediction (OTP) module minimizes temporal redundancy, achieving a TRF score of 0.713 (20.8% higher than baseline VLMs).


## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)
- Other dependencies: See `requirements.txt`

### Step 1: Clone the Repository
```bash
git clone https://github.com/walkstream2025/walkstream.git
cd walkstream
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
# For reminder generation module (GRPO framework)
cd src/reminder_generation
pip install -e .
```

### Step 3: Install Optional Dependencies
- For video processing: `pip install opencv-python ffmpeg-python`
- For evaluation (GPT-Score): `pip install openai`
- For mobile deployment: `pip install coremltools` (iOS) / `pip install torch-mobile` (Android)

## 📊 Dataset Preparation

The Walking Reminder Dataset (WRD) is required for training and evaluation. It includes 98k video clips with annotations for 6 reminder types (obstacle, intersection, road clear/narrow, oncoming vehicle/person, road departure, identifier).

### Download WRD Dataset
```bash
# Navigate to dataset directory
cd src/wrd_dataset
# Run download script (requires wget and unzip)
bash download_wad.sh
# Verify dataset structure (should include video clips, annotations, and metadata)
ls -l WRD/
```

### Dataset Structure
```
WRD/
├── videos/          # 98k annotated video clips (1080p-4K, 60fps)
├── annotations/     # JSON files with reminder type, position, and distance
├── keyframes/       # Extracted 8 keyframes per video (for faster training)
└── metadata.csv     # Dataset statistics (country, scene type, weather, etc.)
```

## 🚀 Quick Start

### 1. Download Pre-trained Checkpoints (Optional)
If you don't want to train from scratch, download pre-trained models:
```bash
cd src/checkpoint
bash download_checkpoint.sh
# Checkpoint will be saved to src/checkpoint/pretrained/
```

### 2. Training

#### Train OTP Module (Temporal Redundancy Reduction)
```bash
cd src/otp
python train.py \
  --dataset_path ../../src/wrd_dataset/WRD \
  --batch_size 8 \
  --epochs 4 \
  --lr 1e-4 \
  --output_dir ../checkpoint/otp_model
```

#### Train Reminder Generation (GRPO Framework)
```bash
cd src/reminder_generation
bash run_grpo_query_gene.sh \
  --dataset_path ../../wrd_dataset/WRD \
  --sft_data_ratio 0.15 \
  --grpo_data_ratio 0.85 \
  --num_epochs 2 \
  --checkpoint_dir ../checkpoint/grpo_model
```

### 3. Inference
Generate real-time walking reminders from video streams (supports webcam, local video, or dataset clips).

```bash
cd src/otp
python inference.py \
  --input_type webcam  # Options: webcam, video, dataset
  --input_path 0       # Webcam ID (0 for default) / path to video file
  --grpo_model_path ../checkpoint/grpo_model/best_model.pth \
  --otp_model_path ../checkpoint/otp_model/best_model.pth \
  --output_dir ../../inference_results \
  --device cuda:0      # Options: cuda, cpu, mps (Mac)
```

### 4. Evaluation
Evaluate model performance on WRD test set (metrics: ROUGE, Keyword Density, GPT-Score, TRF, Latency).

```bash
cd src/otp
python test.py \
  --dataset_path ../../src/wrd_dataset/WRD \
  --grpo_model_path ../checkpoint/grpo_model/best_model.pth \
  --otp_model_path ../checkpoint/otp_model/best_model.pth \
  --gpt_score_path ../infer/GPTScore.py \
  --metrics rouge keyword_density gpt_score trf latency \
  --output_report ../../evaluation_report.csv
```

## 📁 Project Structure
```
walkstream/
├── README.md                # Project documentation
├── LICENSE                  # MIT License
├── requirements.txt         # Core dependencies
├── src/
│   ├── checkpoint/          # Trained model checkpoints
│   │   └── download_checkpoint.sh  # Pre-trained checkpoint download script
│   ├── infer/               # Evaluation tools
│   │   └── GPTScore.py      # GPT-4o based fluency/conciseness evaluation
│   ├── otp/                 # Online Triggered Prediction (OTP) module
│   │   ├── inference.py     # Real-time inference pipeline
│   │   ├── otp_model.py     # OTP core model definition
│   │   ├── test.py          # Evaluation script (OTP + GRPO)
│   │   └── train.py         # OTP module training script
│   ├── reminder_generation/ # GRPO-based reminder generation
│   │   ├── src/             # Core GRPO framework code
│   │   ├── configs/         # Training/inference configurations
│   │   ├── Makefile         # Build script
│   │   ├── README.md        # GRPO module documentation
│   │   ├── run_grpo_query_gene.sh  # GRPO training script
│   │   ├── setup.cfg        # Package configuration
│   │   └── setup.py         # Package installation
│   └── wrd_dataset/         # Dataset download and processing
│       ├── README.md        # Dataset documentation
│       └── download_wad.sh  # Dataset download script
├── inference_results/       # Inference output (auto-generated)
└── evaluation_report.csv    # Evaluation results (auto-generated)
```

## 🎯 Core Modules Explained
| Module | Purpose | Key Files |
|--------|---------|-----------|
| OTP Module | Minimize temporal redundancy by predicting optimal reminder timing | `otp/otp_model.py`, `otp/train.py` |
| GRPO Framework | Reduce output redundancy with 4 reward functions (conciseness/fluency/keyword density/accuracy) | `reminder_generation/run_grpo_query_gene.sh`, `reminder_generation/src/` |
| Inference Pipeline | Real-time video stream processing + reminder generation | `otp/inference.py` |
| Evaluation Tools | Compute quantitative metrics for model performance | `infer/GPTScore.py`, `otp/test.py` |
| Dataset Management | Download and organize WRD dataset | `wrd_dataset/download_wad.sh`, `wrd_dataset/README.md` |
| Checkpoint Management | Download pre-trained models or save training checkpoints | `checkpoint/download_checkpoint.sh` |

## 📈 Experimental Results

### Key Performance Metrics (on NVIDIA A100)
| Metric | WalkStream | Qwen2-VL-3B | GPT-4o | DeepSeekVL2-Small |
|--------|------------|-------------|--------|-------------------|
| ROUGE-1 | 0.467 | 0.420 | 0.145 | 0.431 |
| Keyword Density | 0.455 | 0.414 | 0.263 | 0.447 |
| GPT-Score | 0.843 | 0.814 | - | 0.765 |
| TRF (Temporal Redundancy) | 0.713 | 0.449 | 0.430 | 0.427 |
| Latency (s/frame) | 0.096 | 0.203 | - | 0.790 |

### Real-World Impact
- Reduces manual reminders for BLV users by 37.2% (closed-loop experiment)
- 88.24% positive user feedback on reminder relevance and clarity
- Supports 6 critical walking scenarios with 92% accuracy (obstacle detection)

## 📱 Deployment Options
| Device | Throughput (Token/s) | Latency (s) | Usage |
|--------|----------------------|-------------|-------|
| NVIDIA A100 (40GB) | 200-230 | 0.096 | Server/desktop |
| Mac M4 Pro | 80-100 | 0.15 | Laptop/desktop |
| iPhone A17 Pro | 40-60 | 0.22 | Mobile (smart glasses) |
| Android (Snapdragon 8 Gen 3) | 35-50 | 0.25 | Mobile (smart glasses) |

## 🔧 Troubleshooting
- **Dataset download failures**: Check network connection, or manually download from [dataset link](https://github.com/walkstream2025/walkstream#dataset-access)
- **Training OOM errors**: Reduce batch size (`--batch_size 4`), enable gradient checkpointing (`--gradient_checkpointing True`)
- **Inference latency issues**: Use FP16 precision (`--precision fp16`), or deploy OTP module on CPU
- **GRPO training errors**: Ensure `reminder_generation` is installed in editable mode (`pip install -e .`)
- **Pre-trained checkpoint download failures**: Verify `src/checkpoint/download_checkpoint.sh` has execution permission (`chmod +x download_checkpoint.sh`)

## 🤝 Contributing
We welcome contributions to improve WalkStream! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add unit tests for new features (using `pytest`)
- Update documentation for changes to API or usage
- Reference relevant issues or papers in your PR

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The WRD dataset is licensed under CC BY-NC-SA 4.0 (non-commercial use only). For commercial use, please contact the corresponding author (Jinchao Zhang: jinchao.zhang@tencent.com).

## 🙏 Acknowledgements
- Thanks to YouTube @POPtravel for providing original video data
- Thanks to the BLV participants who contributed to dataset annotation and user testing
- Built on top of Qwen2.5-VL, GRPO, and Detic open-source projects

## 📞 Contact
For questions, issues, or collaboration inquiries:
- Zhiqiang Yuan (first author): yuanzhiqiang19@mails.ucas.ac.cn
- Project GitHub: [https://github.com/walkstream2025/walkstream](https://github.com/walkstream2025/walkstream)

---

Key updates based on your latest directory structure:
1. Updated **Project Structure** to match actual directory hierarchy (e.g., `otp/` module contains `inference.py`/`train.py`/`test.py`, `infer/` only has `GPTScore.py`)
2. Adjusted all command paths (training/inference/evaluation) to point to correct file locations
3. Added **Download Pre-trained Checkpoints** section (using `src/checkpoint/download_checkpoint.sh`)
4. Updated **Core Modules Explained** to map functions to new file paths
5. Maintained consistency with original project goals while ensuring all paths/filenames are accurate

Let me know if you need further adjustments (e.g., add `requirements.txt` content, refine command parameters, or补充 module details)!