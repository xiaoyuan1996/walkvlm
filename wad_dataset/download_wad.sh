#!/usr/bin/env bash
set -euo pipefail

if [ "${PWD##*/}" != "wad_dataset" ]; then
    echo "请先切换到 wad_dataset 目录再执行该脚本"
    exit 1
fi

DATA_URL="https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/seraphyuan/ilabel/blind_vlm/wad_dataset.tar"
ARCHIVE_NAME="wad_dataset.tar"

download_file() {
    local url="$1"
    local output="$2"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$output" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --fail -C - -o "$output" "$url"
    else
        echo "未找到 wget 或 curl，请先安装其中一个。"
        exit 1
    fi
}

echo "下载 WAD 数据集 ..."
download_file "$DATA_URL" "$ARCHIVE_NAME"

echo "解压 WAD 数据集 ..."
tar -zxvf "$ARCHIVE_NAME"

echo "清理压缩包 ..."
rm -f "$ARCHIVE_NAME"

echo "WAD 数据集准备完成。"
