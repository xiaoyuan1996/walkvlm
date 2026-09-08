## 📦 WAD Dataset

The public WAD dataset archive used by this repository can be downloaded with the helper script in this directory.

## ⬇️ Recommended download method

Run the script **inside** `wad_dataset/`:

```bash
cd wad_dataset
bash download_wad.sh
```

The script will:

- download `wad_dataset.tar`
- extract the archive in the current directory
- remove the downloaded tarball after extraction

## ✅ Expected result

After the script finishes, this directory should contain the extracted WAD data used by the maintained `WalkStream` pipeline.

If you are following the recommended reproduction path, continue with:

- root overview: `../README.md`
- maintained pipeline: `../WalkStream/README.md`
