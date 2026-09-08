# WalkStream

WalkStream 是一个面向视障步行辅助场景的视觉语言模型项目，当前仓库主要包含三部分：

- `src/reminder_generation`：基于 GRPO 的提醒生成训练模块
- `src/otp`：OTP（Online Triggered Prediction）时序触发模块
- `src/wrd_dataset`：WRD 数据集下载脚本

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

## 安装

仓库根目录当前**没有统一的 `requirements.txt`**，建议按模块安装：

```bash
pip install torch transformers accelerate datasets pillow scikit-learn opencv-python ffmpeg-python qwen-vl-utils openai rouge-score nltk
cd src/reminder_generation
pip install -e .
```

## 数据集

### 下载 WRD

```bash
cd src/wrd_dataset
bash download_wad.sh
```

### 使用 WAD / WRD 标注训练提醒生成

`src/reminder_generation/run_grpo_query_gene.sh` 现在支持把 `--dataset_path` 传成：

- 数据集根目录
- 或者训练标注文件路径

如果标注里没有 `keyword` / `keywords` 字段，训练脚本会自动从提醒文本中提取关键词。

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

## 已修复的关键问题

- 修复了 `src/wrd_dataset/download_wad.sh` 的目录判断错误
- 修复了 `src/otp` 下脚本的坏导入、空路径占位和导入即运行问题
- 清理了 `src/reminder_generation` 中遗留的私有绝对路径与废弃脚本，避免导入期或运行期误崩溃
- 修复了 `run_grpo_query_gene.sh` 与 README 命令不一致的问题
