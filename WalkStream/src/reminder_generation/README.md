# WalkStream Reminder Generation

`src/reminder_generation` 是 WalkStream 当前**维护中的提醒生成训练入口**，基于 GRPO 对导航提醒进行优化。

## 适用范围

- **推荐用途**：WalkStream 的提醒生成训练与实验复现
- **推荐环境**：Linux + NVIDIA GPU + CUDA + DeepSpeed
- **不再推荐**：旧版 `WalkVLM-LR` 模板脚本和历史副本脚本

## 功能说明

- 支持通过 `run_grpo_query_gene.sh` 启动训练。
- `--dataset_path` 既可以传**数据集根目录**，也可以直接传**训练标注文件**。
- 如果标注中没有 `keyword` / `keywords` 字段，脚本会从 `reminder` 文本中自动提取关键词，避免旧版数据格式报错。
- 如果关键帧目录中没有 `cache_qwen2vl.pt`，脚本会自动回退到随机 cache，不再强依赖私有缓存文件。

## 安装

```bash
cd WalkStream/src/reminder_generation
pip install -e .
```

如果你还没有创建虚拟环境，建议先在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

## 最小 smoke test

安装完成后，先确认训练入口脚本本身可见：

```bash
cd WalkStream/src/reminder_generation
bash run_grpo_query_gene.sh --help
```

如果你能看到帮助信息或参数说明，说明包装脚本已经能被正确解析。

## 数据输入约定

训练样本至少应满足以下条件：

- 标注文件中存在 `frame_path`
- 标注文件中存在 `reminder` 或兼容字段（如旧数据中的 `alter`）
- 对应关键帧目录下能找到 `8.jpg` / `8.jpeg` / `8.png` 这类图片

`--dataset_path` 可以是两种形式：

- **数据集根目录**：脚本会自动尝试查找训练 / 验证标注与图片目录
- **标注文件路径**：脚本会以该文件为主，并结合 `--image_root` / `--eval_image_root` 解析图片

## 训练

最常用的启动方式如下：

```bash
cd WalkStream/src/reminder_generation
bash run_grpo_query_gene.sh \
  --dataset_path ../../wad_dataset \
  --output_dir ../checkpoint/grpo_model \
  --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
  --num_train_epochs 2 \
  --run_name walkstream-grpo
```

## 可选参数

- `--dataset_path`：数据集根目录，或训练标注文件路径。
- `--dataset_val_path`：验证标注文件路径；不传时自动尝试搜索 `test` / `val` / `eval` 标注文件。
- `--image_root`：训练集关键帧根目录；不传时自动从常见目录名中搜索。
- `--eval_image_root`：验证集关键帧根目录；不传时默认复用训练集目录。
- `--cache_filename`：每个样本旁的 cache 文件名，默认 `cache_qwen2vl.pt`。
- `--disable_cache`：禁用 cache 文件读取，统一回退到随机 cache。

## 环境变量

如果你希望复用本地缓存好的奖励模型，可设置：

```bash
export WALKSTREAM_GPT2_MODEL=/path/to/gpt2
export WALKSTREAM_CLIP_MODEL=/path/to/clip-vit-base-patch16
```

如果不设置，脚本会回退到公开可下载模型。

## 兼容性说明

以下旧参数**并不被当前训练入口使用**，不要再写进命令里：

- `--sft_data_ratio`
- `--grpo_data_ratio`
- `--checkpoint_dir`

如果你需要把训练输出保存到指定目录，请使用 `--output_dir`。
