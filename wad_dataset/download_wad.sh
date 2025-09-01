#!/bin/bash

# 检查当前所处路径
if [ "${PWD##*/}" != "wad_dataset" ]; then
    echo "Switch to the wad_dataset folder to unzip"
    exit 1
fi


# 下载数据集
echo " Downloading the WAD dataset .."
wget https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/seraphyuan/ilabel/blind_vlm/wad_dataset.tar

# 解压
echo " Untar WAD dataset .."
tar -zxvf wad_dataset.tar

# 删除缓存
echo "Delete the WAD archive .."
rm -rf wad_dataset.tar

echo " WAD dataset processed."