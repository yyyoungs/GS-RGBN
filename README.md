# GS-RGBN
[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_High-fidelity_3D_Object_Generation_from_Single_Image_with_RGBN-Volume_Gaussian_CVPR_2025_paper.pdf) | [Webpage](https://gapszju.github.io/GS-RGBN/)

Official implementation of CVPR2025 paper "High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model".

## Installation

1. Clone this repository.

   ```
   git clone https://github.com/yyyoungs/GS-RGBN.git
   cd GS-RGBN
   ```

2. Create conda environment.

   ```
   conda create -n GS-RGBN python=3.9
   conda activate GS-RGBN
   ```

3. Install [PyTorch](https://pytorch.org/get-started/locally/). Please make sure that the PyTorch CUDA version matches your system's CUDA version. We use CUDA 12.1 here.

   ```
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
   ```

4. Install other packages.

   ```
   pip install -r requirements.txt
   ```

5. Install [2D Gaussian Splatting (2DGS)](https://github.com/hbb1/2d-gaussian-splatting).

   ```
   git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
   cd 2d-gaussian-splatting
   pip install submodules/diff-surfel-rasterization
   pip install submodules/simple-knn
   ```

### Dataset

1. We render [Objaverse](https://objaverse.allenai.org/objaverse-1.0) and [GSO](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) for train sets and test sets, respectively. The rendering codes please refer to [kiuikit](https://github.com/ashawkey/kiuikit) or [Wonder3D](https://github.com/xxlong0/Wonder3D).
2. We use [Wonder3D](https://github.com/xxlong0/Wonder3D) or [Wonder3D++](https://github.com/xxlong0/Wonder3D/tree/Wonder3D_Plus) to generate corresponding multi-view diffusion (MVD) images for training and testing. 
3. (Optional) Organize all the files as follows:
```
data/dataset_name
|-- 000
    |-- object1
       |-- 000 (0 azimuth degree)
          |-- 000
             |-- gt.png (GT RGB images)
             |-- gt_depth.png/exr (GT depth images)
             |-- mv_rgb.png (Generated RGB images by Wonder3D)
             |-- mv_normal.png (Generated normal images by Wonder3D)
          |-- 001
          |-- 002
          ...
       |-- 090 (90 azimuth degree)
       ...
    |-- object2
    |-- object3
    ...
...
```

### Training

```
# single gpu training
python single_gpu_train.py

# multiple training
accelerate launch --config_file configs/acc_config.yaml multiple_gpu_train.py
```
### Inference
```
# Custom images
python infer_custom.py
```

### Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)
- [LGM](https://github.com/3DTopia/LGM)
- [Lara](https://github.com/autonomousvision/LaRa)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)

### Citation

```
@inproceedings{shen2025high,
  title={High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model},
  author={Shen, Yiyang and Zhou, Kun and Wang, He and Yang, Yin and Shao, Tianjia},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={21558--21569},
  year={2025}
}

```




