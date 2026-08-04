#!/bin/bash

# 检查当前所处路径
if [ "${PWD##*/}" != "wad_dataset" ]; then
    echo "应切换到 wad_dataset 文件夹下解压"
    exit 1
fi


# 下载数据集
echo " WAD 数据集下载 .."
wget https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/seraphyuan/ilabel/blind_vlm/WRD.zip

echo " WAD 测试数据集下载 .."
wget https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/seraphyuan/ilabel/blind_vlm/WRD_test.zip

# 解压
echo " WAD 数据集解压 .."
unzip WRD.zip
unzip WRD_test.zip

# 删除缓存
echo "删除 WAD 压缩包 .."
rm -rf WRD.zip
rm -rf WRD_test.zip

echo "数据集处理完毕"