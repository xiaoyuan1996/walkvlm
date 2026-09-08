## WRD 数据集

`download_wad.sh` 会下载 WalkStream 使用的 WRD 训练集和测试集压缩包，并在当前目录中自动解压。

## 下载方法

请先切换到 `wrd_dataset/` 目录：

```bash
cd WalkStream/src/wrd_dataset
bash download_wad.sh
```

## 脚本行为

脚本会依次：

- 下载 `WRD.zip`
- 下载 `WRD_test.zip`
- 解压两个压缩包
- 删除下载得到的 zip 文件

## 预期结果

执行完成后，当前目录下应出现解压后的 WRD 数据目录，可供 `WalkStream` 训练和评测流程使用。

如果你需要旧版 WAD 数据，请使用仓库根目录下的 `wad_dataset/download_wad.sh`。
