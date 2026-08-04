#!/bin/bash
set -euo pipefail

# 检查当前所处路径
if [ "${PWD##*/}" != "checkpoint" ]; then
    echo "应切换到 checkpoint 文件夹下执行"
    exit 1
fi

BASE_URL="https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/leonchadli/checkpoint"

FILES=(
    "added_tokens.json"
    "chat_template.jinja"
    "config.json"
    "generation_config.json"
    "merges.txt"
    "model.safetensors"
    "preprocessor_config.json"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "trainer_state.json"
    "training_args.bin"
    "video_preprocessor_config.json"
    "vocab.json"
)

download_file() {
    local file="$1"
    local url="${BASE_URL}/${file}"

    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$file" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --fail -C - -o "$file" "$url"
    else
        echo "未找到 wget 或 curl，请先安装其中一个"
        exit 1
    fi
}

echo "checkpoint 下载中 .."

for file in "${FILES[@]}"; do
    echo "下载: $file"
    download_file "$file"
done

echo "checkpoint 下载完毕"
