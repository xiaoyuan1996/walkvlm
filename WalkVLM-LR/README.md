# WalkVLM-LR

A walking assistance VLM with reduced redundancy for blind and low vision individuals.

---

## 📖 Overview

**WalkVLM-LR** is a walking assistance model designed to improve navigation for individuals with visual impairments.  

✨ Key features:  
- Reduces both **output** and **temporal redundancy** compared to existing models.  
- Uses **human-preference-based custom reward functions** and an **environment awareness discriminator (EAD)**.  
- Generates **concise, accurate, and context-appropriate guidance**, minimizing unnecessary reminders.  

📊 Experimental results demonstrate that WalkVLM-LR outperforms other models in:  
- **Output conciseness**  
- **Reducing temporal redundancy**  

---

## ⚙️ Installation

### Prerequisites
- CUDA **12.4**
- Python **3.11**
- PyTorch **2.6.0**

### Setup

```bash
# Clone the repository
git clone https://github.com/huggingface/open-r1.git

# Install dependencies
pip install --upgrade pip
pip install vllm==0.8.5.post1
pip install setuptools && pip install flash-attn --no-build-isolation
GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"
```
## 📂 Dataset & Pretrained Weights
Before training or inference, configure the dataset and download pretrained weights:
```bash
# Download the dataset
cd checkpoint
bash download_wad.sh

# Download pretrained weights
cd checkpoint
bash download_checkpoint.sh
```
The dataset will be placed in wad_dataset/ and pretrained weights in checkpoint/.

## 🎯 Training

We use GRPO to fine-tune WalkVLM-LR.

### GRPO Training Command

```bash
# Run the training script
cd vlm_grpo_template
bash run_grpo_query_gene.sh
```

### GRPO Training Configuration

You can modify the training parameters in the `run_grpo_query_gene.sh` script, including:
- output_dir
- max_prompt_length
- num_train_epochs
- dataset_path 

### 🧩 EAD Training Command
Configure the dataset path inside train_EAD.py, then run:
```bash
python -m torch.distributed.launch --nproc_per_node=8 train_EAD.py
```

## 🧪 Testing Command
Modify the checkpoint_path and image_paths in test.py, and then proceed with the testing. The pre-trained weights for CLIP and the related parameters for the GPT-4 API need to be configured manually.
```bash
python test.py
```
⚠️ The pretrained weights for CLIP and parameters for the GPT-4 API need to be configured manually.

## 🚀 Inference Command
Modify the checkpoint_path and image_paths (with a limit of three images per input) in inference.py, and then perform the inference.

```bash
python inference.py
```

## 📬 Contact

For questions and feedback, please open an issue and contact.
