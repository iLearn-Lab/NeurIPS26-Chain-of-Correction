<div align="center">

<h1>Chain-of-Correction: Progress-Aware Policy Steering via Anchor-Grounded Predictive Reasoning</h1>

<div>
  <a target="_blank" href="https://scholar.google.com/citations?user=ORL0j-AAAAAJ">Xuening&#160;Zhang</a><sup>1</sup>, &nbsp;
  Xiang&#160;Deng<sup>1&#9993;</sup>, &nbsp;
  Jiayi&#160;Lin<sup>1</sup>, &nbsp;
  <a target="_blank" href="https://aopolin-lv.github.io/">Qi&#160;Lv</a><sup>1</sup>, &nbsp;
  Xingbo&#160;Liu<sup>2</sup>, &nbsp;
  Weili&#160;Guan</a><sup>1</sup>
</div>

<sup>1</sup>Harbin Institute of Technology (Shenzhen) &nbsp;&nbsp;&nbsp;
<sup>2</sup>Shandong Jianzhu University
<br />
<sup>&#9993;</sup>Corresponding Author

<div align="center">
  <a href="https://arxiv.org/pdf/2603.21051" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper"></a>
  <a href="https://github.com/zxnlingling/ICLR26-Cortical_Policy/releases/tag/pretrained-GLC" target="_blank">
  <img src="https://img.shields.io/badge/Pretrained-Gaze--Model-green" alt="Gaze Model"></a>
  <a href="assets/Cortical_Policy_ICLR2026.png" target="_blank">
    <img src="https://img.shields.io/badge/Poster-ICLR--2026-blue" alt="Poster"></a>
</div>

</div>


## Updates
- [10/2026] Codebase released. Check out our [Installation Guide](#️-installation).
- [09/2026] Chain-of-Correction has been accepted to **NeurIPS 2026**.


## 🌟 Overview

<p align="center">
  <img src="assets/spatial_task.webp" width="45%" />
  <img src="assets/dynamic_task.webp" width="45%" />
  <br>
  <em>Cortical Policy enhances spatial reasoning (left) and dynamic adaptation (right) via a dual-stream architecture.</em>
</p>

<p align="center">
  <img src="assets/Framework_CP.png" width="75%" />
  <br>
  <em>Framework of Cortical Policy.</em>
</p>

**Key Features:**
* **Dual-Stream Perception**: Inspired by the human visual cortex, it integrates a **static-view stream** and a **dynamic-view stream** for comprehensive perception.
* **Geometric Consistency**: Enforces cross-view feature alignment via **VGGT** for high-precision manipulation.
* **Egocentric Gaze Prior**: Pretrains a gaze model to extract action-oriented representations from dynamic views. 

## 📊 Experimental Results

### Results on RLBench
Cortical Policy outperforms SOTA view transformers across 18 diverse manipulation tasks.

| Method | Reference | Avg. Success ↑ | Insert Peg | Put in Cupboard | Stack Cups | Close Jar |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| PerAct | CoRL 2022 | 49.4% | 5.6% | 28.0% | 2.4% | 55.2% |
| RVT | CoRL 2023 | 62.9% | 11.2% | 49.6% | 26.4% | 52.0% |
| RVT-2 | RSS 2024 | 77.5% | 28.0% | 44.0% | 72.0% | 93.3% |
| 3D-MVP | CVPR 2025 | 67.5% | 20.0% | 60.0% | 36.0% | 76.0% |
| **Ours** | **ICLR 2026** | **81.0%** | **38.7%** | **65.3%** | **76.0%** | **96.0%** |

> *For the complete 18-task performance, please refer to our [Full Results Table](assets/rlbench_results.png).*

### Results on THE COLOSSEUM
Cortical Policy demonstrates strong robustness against **environmental perturbations** (e.g., unseen distractors, lighting changes, and camera pose shifts). This capability is primarily driven by our **dynamic-view stream**.

<p align="center">
  <img src="assets/colosseum_results.png" width="90%" />
</p>

> **Ablation Variants:**
> - **Variant B**: Uses only the static-view stream.
> - **Variant D**: Employs dynamic-view feature maps without heatmaps, and removes the static-view geometric consistency loss ($\mathcal{L}_{cgc}$).
> - **Variant E**: Retains the full dynamic-view stream but removes $\mathcal{L}_{cgc}$.

### Results on Real-World Tasks
Cortical Policy is deployed on the AgileX ALOHA robotic platform, utilizing two fixed cameras for static-view perception and a wrist-mounted camera for dynamic-view perception. Our method demonstrates superior spatial reasoning and dynamic adaptation under physical world constraints.

<p align="center">
  <img src="assets/real_results.png" width="60%">
  <br>
  <span style="color: #666; font-size: 0.9em;">
    <em><strong>Real-world evaluation across four setups:</strong> a basic task ("stack 2 blocks"), a spatial reasoning task ("stack 2 blocks in between bottles"), and two dynamic tasks (target/base displacement).</em>
  </span>
</p>

## 🛠️ Installation
We recommend using **Python 3.10** and **CUDA 11.8**.
```bash
conda create -n cortical python=3.10 -y
conda activate cortical

# 1. Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 2. Install basic requirements
pip install -r requirements.txt
```
Our framework relies on customized simulators and renderers. Before proceeding, **you must install CoppeliaSim** (V4.1.0) and set up the required environment variables (e.g., `COPPELIASIM_ROOT`). Please refer to the [PyRep Official Guide](https://github.com/stepjam/PyRep) for detailed instructions.

Once CoppeliaSim is set up, specify your CUDA path and install the local libraries:
```bash
# 3. Setup CUDA path for building extensions (Add these to your ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-11.8 
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 4. Build and install local libraries
pip install -e libs/pytorch3d
pip install -e libs/point-renderer
pip install -e libs/PyRep
pip install -e libs/RLBench
pip install -e libs/YARR
pip install -e libs/robot-colosseum
```

## 📂 Data & Model Preparation
### 1. Generate Training Data
This pipeline extracts geometrically-consistent keypoints (i.e., 3D supervision) from raw RLBench demonstrations and prepares replay samples for training the dual-stream architecture. 

Ensure your data folder is organized as follows:
```bash
your_data_folder/
└── train/
    └── your_task_name/
        └── all_variations/
            └── episodes/
                ├── episode0/
                ├── episode1/
                ...
```
Then, run the data preparation script:
```bash
python scripts/replay_preparation.py \
  --exp_cfg_path configs/cp.yaml \
  --data_folder /path/to/your/rlbench/demos
```
> *Note: By default, we use 100 demos per task, following the exact setup in our paper.*

### 2. Pretrain Gaze Model (Dynamic-View Stream)
First, download the initial [Ego4D-pretrained weights](https://drive.google.com/file/d/1zQdZuV4OowEJxBZAAWr_MuZcN7LJujOw/view?usp=sharing) and place them in `src/GLC/checkpoints/MViT_Ego4D_ckpt.pyth`.

Next, run the following command to pretrain the dynamic-view backbone:
```bash
CUDA_VISIBLE_DEVICES=0 python src/GLC/pretrain_GLC.py \
  --cfg src/GLC/configs/MVIT_B_16x4_CONV.yaml \
  TRAIN.BATCH_SIZE 16 TEST.ENABLE False NUM_GPUS 1 \
  TRAIN.CHECKPOINT_FILE_PATH src/GLC/checkpoints/MViT_Ego4D_ckpt.pyth \
  OUTPUT_DIR src/GLC/checkpoints/
```
The final pretrained weights will be saved to `src/GLC/checkpoints/checkpoints/checkpoint_epoch_00015.pyth`. 
*(Alternatively, you can directly download our fully pretrained GLC weights from [GitHub Releases](https://github.com/zxnlingling/ICLR26-Cortical_Policy/releases/download/pretrained-GLC/checkpoint_epoch_00015.pyth)).*

## 🚀 Training Cortical Policy
Once the data is prepared and the gaze model is pretrained, you can start the policy training. 
```bash
export PYTHONPATH=$PYTHONPATH:.
python scripts/train.py --exp_cfg_path configs/cp.yaml --mvt_cfg_path src/mvt/configs/rvt2.yaml --device 0,1
```

## ⚖️ Evaluate Cortical Policy
*(Coming soon...)*

## 🙏 Acknowledgements
This work builds upon [RVT-2](https://github.com/NVlabs/RVT), [VGGT](https://github.com/facebookresearch/vggt), and [GLC](https://github.com/BolinLai/GLC). We thank the authors for open-sourcing their excellent codebases.

## 📝 Citation
If you find this work useful in your research, please cite our paper:
```bibtex
@inproceedings{zhang2026cortical,
  title={Cortical Policy: A Dual-Stream View Transformer for Robotic Manipulation},
  author={Xuening Zhang and Qi Lv and Xiang Deng and Miao Zhang and Xingbo Liu and Liqiang Nie},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## ✉️ Contact
If you have any questions or find any bugs, please feel free to open an issue or contact [Xuening Zhang](mailto:yukiZhang0527@outlook.com).
