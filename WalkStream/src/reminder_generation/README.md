# WalkStream Reminder Generation

`src/reminder_generation` 是 WalkStream 的提醒生成训练模块，基于 GRPO 对导航提醒进行优化。

## 功能说明

- 支持通过 `run_grpo_query_gene.sh` 启动训练。
- `--dataset_path` 既可以传**数据集根目录**，也可以直接传**训练标注文件**。
- 如果标注中没有 `keyword` / `keywords` 字段，脚本会从 `reminder` 文本中自动提取关键词，避免旧版数据格式报错。
- 如果关键帧目录中没有 `cache_qwen2vl.pt`，脚本会自动回退到随机 cache，不再强依赖私有缓存文件。

## 安装

```bash
cd src/reminder_generation
pip install -e .
```

## 训练

最常用的启动方式如下：

```bash
cd src/reminder_generation
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

## 兼容性说明

以下旧参数**并不被训练入口使用**，不要再写进命令里：

- `--sft_data_ratio`
- `--grpo_data_ratio`
- `--checkpoint_dir`

如果你需要把训练输出保存到指定目录，请使用 `--output_dir`。
