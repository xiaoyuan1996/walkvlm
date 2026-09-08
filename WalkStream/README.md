# WalkStream

WalkStream 是当前仓库中**推荐用于复现**的主路径，面向视障步行辅助场景，主要包含三部分：

- `src/reminder_generation`：基于 GRPO 的提醒生成训练模块
- `src/otp`：OTP（Online Triggered Prediction）时序触发模块
- `src/wrd_dataset`：WRD 数据集下载脚本

## 当前支持状态

| 模块 | 状态 | 适合新用户直接使用？ | 说明 |
|---|---|---|---|
| `src/reminder_generation` | 维护中 | **是** | 当前推荐的提醒生成训练入口 |
| `src/otp` | 维护中 | **是** | 当前推荐的 OTP 训练 / 推理 / 评测入口 |
| `src/wrd_dataset` | 维护中 | **是** | WRD 数据下载入口 |
| `../WalkVLM-LR` | 已归档 | 否 | 仅保留作历史研究快照 |

## 环境基线

为了提高复现成功率，建议优先使用以下环境：

- **操作系统**：Linux
- **Python**：3.10 或 3.11
- **GPU**：NVIDIA GPU
- **训练提醒生成模块时**：需要 CUDA + DeepSpeed

说明：

- `src/otp` 下的 CLI 帮助和轻量检查不强依赖 DeepSpeed。
- `src/reminder_generation` 的完整训练链路默认面向 Linux + CUDA 环境；macOS / Windows 不作为主要训练目标。

## 安装

仓库根目录当前没有统一的顶层 `requirements.txt`，建议以 `WalkStream/src/reminder_generation` 的 editable install 作为主安装入口：

```bash
cd walkvlm
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

cd WalkStream/src/reminder_generation
pip install -e .
```

这一步现在会同时安装提醒生成、OTP 推理与评测脚本所需的公共依赖，例如 `qwen-vl-utils`、`nltk`、`rouge-score`、`scikit-learn`、`openai`、`requests` 等。

## 最小 smoke test

在不下载大模型和数据之前，建议先确认入口脚本能被正确解析：

```bash
cd WalkStream/src/otp
python train.py --help
python inference.py --help
python test.py --help

cd ../reminder_generation
bash run_grpo_query_gene.sh --help
```

如果上面的命令能输出 `usage` 或帮助信息，说明基础 CLI 和依赖安装已经基本正确。

## 目录结构

```text
WalkStream/
├── README.md
└── src/
    ├── checkpoint/
    │   └── download_checkpoint.sh
    ├── infer/
    │   └── GPTScore.py
    ├── otp/
    │   ├── inference.py
    │   ├── otp_model.py
    │   ├── test.py
    │   └── train.py
    ├── reminder_generation/
    │   ├── README.md
    │   ├── run_grpo_query_gene.sh
    │   └── src/
    └── wrd_dataset/
        ├── README.md
        └── download_wad.sh
```

## 数据与权重准备

### 1. 下载 WRD

```bash
cd src/wrd_dataset
bash download_wad.sh
```

WRD 下载脚本会自动下载并解压训练集、测试集压缩包。

### 2. 下载 WAD

如果你需要旧版 WAD 数据或依赖 `wad_dataset` 目录布局的流程，请使用仓库根目录下的脚本：

```bash
cd ../wad_dataset
bash download_wad.sh
```

### 3. 下载可选 checkpoint

如果你想先复用仓库提供的 checkpoint 文件，可执行：

```bash
cd src/checkpoint
bash download_checkpoint.sh
```

## 训练

### 1. 训练 OTP 分类器

`src/otp/train.py` 现在接收显式参数，不再依赖空的占位路径：

```bash
cd src/otp
python train.py \
  --data_dir /path/to/otp_features \
  --output_path ../checkpoint/otp_model.pth \
  --epochs 4 \
  --batch_size 8 \
  --lr 1e-4
```

说明：`--data_dir` 应指向包含 `*_features.json` 的目录树。

### 2. 训练提醒生成模块

```bash
cd src/reminder_generation
bash run_grpo_query_gene.sh \
  --dataset_path ../../wad_dataset \
  --output_dir ../checkpoint/grpo_model \
  --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
  --num_train_epochs 2 \
  --run_name walkstream-grpo
```

如需使用本地缓存好的奖励模型，可通过环境变量指定：

```bash
export WALKSTREAM_GPT2_MODEL=/path/to/gpt2
export WALKSTREAM_CLIP_MODEL=/path/to/clip-vit-base-patch16
```

## 推理

```bash
cd src/otp
python inference.py \
  --model_name_or_path /path/to/grpo_or_vlm_checkpoint \
  --classifier_checkpoint ../checkpoint/otp_model.pth \
  --image_paths /path/to/frame1.jpg /path/to/frame2.jpg /path/to/frame3.jpg
```

- 若不提供 `--classifier_checkpoint`，脚本会直接对最后一帧生成提醒文本。
- 若提供分类器，则当 OTP 预测结果包含 `C` 时才生成提醒；也可以加 `--always_generate` 强制生成。

## 评测

```bash
cd src/otp
python test.py \
  --model_name_or_path /path/to/grpo_or_vlm_checkpoint \
  --dataset_path /path/to/test.jsonl \
  --image_root /path/to/src_data \
  --output_file ../../evaluation_report.json
```

可选地传入 `--gpt_appid`、`--gpt_appkey`、`--gpt_source` 以启用 GPT 评分；若不传，则仅计算 ROUGE 和关键词密度。

## 已知限制

- `src/reminder_generation` 仍依赖 DeepSpeed / Accelerate 训练栈，完整训练更适合 Linux + CUDA 环境。
- GPT 评分依赖外部接口与凭据，因此默认不是最小复现路径的一部分。
- 旧版 `WalkVLM-LR` 与模板目录已归档，不应再作为主复现入口。

## 已修复的关键问题

- 修复了 `src/wrd_dataset/download_wad.sh` 的目录判断错误
- 修复了 `src/otp` 下脚本的坏导入、空路径占位和导入即运行问题
- 清理了 `src/reminder_generation` 中遗留的私有绝对路径与废弃脚本，避免导入期或运行期误崩溃
- 修复了 `run_grpo_query_gene.sh` 与 README 命令不一致的问题
- 补齐了 `pip install -e .` 覆盖不到的关键运行时依赖
