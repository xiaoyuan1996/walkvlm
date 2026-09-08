#!/usr/bin/env bash
set -euo pipefail

if [ "${PWD##*/}" != "wrd_dataset" ]; then
    echo "请先切换到 wrd_dataset 目录再执行该脚本"
    exit 1
fi

WRD_URL="https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/seraphyuan/ilabel/blind_vlm/WRD.zip"
WRD_TEST_URL="https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/seraphyuan/ilabel/blind_vlm/WRD_test.zip"

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

echo "下载 WRD 训练集 ..."
download_file "$WRD_URL" "WRD.zip"

echo "下载 WRD 测试集 ..."
download_file "$WRD_TEST_URL" "WRD_test.zip"

echo "解压 WRD 数据 ..."
unzip -o WRD.zip
unzip -o WRD_test.zip

echo "清理压缩包 ..."
rm -f WRD.zip WRD_test.zip

echo "WRD 数据集准备完成。"
