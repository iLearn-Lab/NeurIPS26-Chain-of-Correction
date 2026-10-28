<div align="center">

<h1>Chain-of-Correction: Progress-Aware Policy Steering<br>via Anchor-Grounded Predictive Reasoning</h1>

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
  <a href="" target="_blank">
    <img src="https://img.shields.io/badge/Paper-coming%20soon-red" alt="Paper: coming soon"></a>
  <a href="" target="_blank">
  <img src="https://img.shields.io/badge/Checkpoint-green" alt="CoC Model"></a>
  <a href="" target="_blank">
    <img src="https://img.shields.io/badge/Poster-coming%20soon-blue" alt="Poster: coming soon"></a>
</div>

</div>


## Updates
- [10/2026] Codebase released. Check out our [Installation Guide](#️-installation).
- [09/2026] Chain-of-Correction has been accepted to **NeurIPS 2026**.


## 🌟 Overview

Imitation learning policies can accumulate small execution errors into task failures. **Chain-of-Correction (CoC)** evaluates each proposed action before execution, using anchor-grounded scene graphs and explicit task progress to decide whether to retain the proposal or synthesize a corrective action.

<p align="center">
  <img src="assets/spatial_task.webp" width="45%" />
  <img src="assets/dynamic_task.webp" width="45%" />
  <br>
  <em>CoC steers a frozen manipulation policy at inference time through a single vision-language model.</em>
</p>

<p align="center">
  <img src="assets/Framework_CP.png" width="75%" />
  <br>
  <em>Framework of Chain-of-Correction (CoC).</em>
</p>

**Key Features:**
* **Anchor-grounded scene graphs** focus on the gripper, manipulation object, and target, with spatial relations and continuous geometric attributes.
* **Progress-aware dual-check**: A backward check identifies historical execution errors. A forward check assesses the proposed action's consequences for task completion.
* **Unified failure-augmented SFT** distill scene graph extraction, progress estimation, and action correction into one VLM initialized from MomaGraph-R1. 

## 📊 Experimental Results

### Results on THE COLOSSEUM
Cortical Policy outperforms SOTA view transformers across 18 diverse manipulation tasks.

| Method | Reference | Avg. Success ↑ | All Perturbations ↑ | MO Color ↑ | Light Color ↑ | Distractor Objects ↑ | MO Size ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BridgeVLA (base policy) | NeurIPS 2025 | 66.7% | 24.8% | 58.5% | 66.1% | 68.6% | 74.5% |
| FOREWARN | RSS 2025 | 50.8% | 67.4% | 68.0% | 45.7% | 36.0% | 70.2% |
| LPB | NeurIPS 2025 | 66.9% | 25.3% | 56.9% | 62.5% | 70.7% | 77.1% |
| GPC | RA-L 2026 | 67.6% | 27.8% | 60.2% | 65.9% | 70.7% | 78.2% |
| **CoC** | **Ours** | **71.6%** | **67.6%** | **70.9%** | **68.4%** | **71.2%** | **78.5%** |

### Offline Predictive Reasoning Results
We evaluate CoC on a balanced 200-sample offline validation set, constructed by pooling the failure-augmented training set and randomly sampling exactly 100 positive instances requiring correction and 100 negative instances requiring no intervention. CoC outperforms zero-shot general VLMs and domain-specific embodied VLMs in error detection and progress estimation, achieving an **error-detection F1 score of 0.851**, **precision of 1.000**, **recall of 0.740**, and a **progress MAE of 0.007**.

| Method | F1 Score ↑ | Precision ↑ | Recall ↑ | Progress RMSE ↓ | Progress MAE ↓ | Trans. L2 ↓ | Rot. L2 ↓ | Grip. Err. ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B-Instruct | 0.397 | 0.683 | 0.280 | 0.126 | 0.072 | 0.043 | 0.546 | 21.0% |
| Llama-3.2-11B-Vision-Instruct | 0.539 | 0.496 | 0.590 | 0.359 | 0.242 | 0.085 | 0.542 | 22.8% |
| GPT-5.4 | 0.546 | 0.622 | 0.487 | 0.070 | 0.049 | 0.155 | 0.553 | 28.0% |
| MomaGraph-R1 | 0.085 | 0.278 | 0.050 | 0.086 | 0.047 | 0.030 | 0.478 | **20.5%** |
| VLM-3R | 0.472 | 0.475 | 0.470 | 0.335 | 0.230 | 0.084 | 0.470 | 25.6% |
| **CoC** | **0.851** | **1.000** | **0.740** | **0.037** | **0.007** | **0.020** | **0.121** | 23.0% |

### Real-World Error Recovery
CoC is deployed on the AgileX ALOHA robotic platform, and demonstrates precise geometric reasoning, robust generalization to visual perturbations, and progress-aware error recovery in long-horizon manipulation under real-world physical constraints.

<p align="center">
  <img src="assets/real_world_results.png" width="90%" alt="Real-world evaluation of CoC">
  <br>
  <span style="color: #666; font-size: 0.9em;">
    <em><strong>Real-world evaluation across four setups:</strong> a basic precise manipulation task ("Move Hanger"), two visual generalization settings (unseen object color and distractor objects), and a long-horizon sequential task ("Pack into Box").</em>
  </span>
</p>

## 🛠️ Installation
We recommend using separate environments for CoC and the base policy. Take BridgeVLA as an example:

| Environment | Python | PyTorch | Transformers |
| :--- | :---: | :---: | :---: |
| CoC VLM (`environment.yml`) | 3.10 | 2.4.1 | 5.3.0 |
| BridgeVLA (`bridgevla_env_backup.yml`) | 3.9 | 2.5.1+cu121 | 4.51.3 |

### 1. Create the VLM environment
```bash
conda env create -f environment.yml -n cot_sft
conda activate cot_sft
export COC_ROOT="$(pwd)"
export PYTHONPATH="$COC_ROOT:${PYTHONPATH:-}"
```

### 2. Prepare the policy environment
Use `bridgevla_env_backup.yml` as the source environment reference; remove its local `prefix:` before recreating it. 

Install **CoppeliaSim 4.1.0**, following the [PyRep installation guide](https://github.com/stepjam/PyRep). The supplied code package contains a CoppeliaSim directory at the following default location:

```bash
export COPPELIASIM_ROOT="$COC_ROOT/policy/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT/platforms"
```

Populate the  `bridgevla/libs/PyRep` and `bridgevla/libs/RLBench` directories using the installation scripts:
```bash
conda activate bridgevla
cd "$COC_ROOT/policy/BridgeVLA/finetune/bridgevla/libs"
git clone https://github.com/stepjam/PyRep.git PyRep
git -C PyRep checkout 231a1ac6b0a179cff53c1d403d379260b9f05f2f
git clone https://github.com/buttomnutstoast/RLBench.git RLBench
git -C RLBench checkout 587a6a0e6dc8cd36612a208724eb275fe8cb4470

cd "$COC_ROOT/policy/BridgeVLA/finetune"
pip install -e .
pip install -e bridgevla/libs/PyRep
pip install -e bridgevla/libs/RLBench
pip install -e bridgevla/libs/YARR
pip install -e bridgevla/libs/peract_colab
pip install -e bridgevla/libs/point-renderer
pip install -r Colosseum/robot-colosseum/requirements.txt
pip install -e Colosseum/robot-colosseum
cp Colosseum/utils/colosseum_utils.py bridgevla/libs/RLBench/rlbench/utils.py
```

### 3. Set local paths and obtain the initial model weights
Download [MomaGraph-R1](https://huggingface.co/cheryyunl/MomaGraph-R1) for the VLM initialization. Prepare the BridgeVLA policy weights and the PaliGemma backbone used by that policy, following its upstream documentation. A policy checkpoint directory must contain `model_<epoch>.pth`, `exp_cfg.yaml`, and `mvt_cfg.yaml`.

Before running the commands below, configure the source paths including:

- `configs/sft_config.yaml`: `model.path`.
- `vlm_corrector_server.py`: the default `base_model_path` in `VLMCorrector.__init__`.
- `policy/BridgeVLA/finetune/bridgevla/mvt/mvt_single.py`: `model_id` for PaliGemma.
- `policy/BridgeVLA/finetune/Colosseum/eval_worker.sh`: repository and test-data paths.

## 📂 Data Preparation
### 1. Organize demonstrations and failure rollouts
Ensure your data folder is organized as follows:
```bash
/data/colosseum_dataset/
├── train/<task>/all_variations/episodes/episode0/
│   ├── low_dim_obs.pkl
│   ├── variation_descriptions.pkl
│   ├── front_rgb/0.png
│   ├── left_shoulder_rgb/0.png
│   ├── right_shoulder_rgb/0.png
│   └── wrist_rgb/0.png
├── failed_eval_demos/<task>/<task>_<variation>/variation0/episodes/episode0/
│   ├── low_dim_obs.pkl
│   ├── variation_descriptions.pkl
│   └── ... camera images ...
└── test/<task>/<task>_<variation>/variation0/episodes/episode0/
    └── ... evaluation demonstrations ...
```
We use **100 expert demonstrations per task** and collects failures from a frozen BridgeVLA policy. To collect new failures with this code, set `vlm_corrector = None` at the marked assignment in `Colosseum/eval.py`, run that evaluator with `--save_failed`, and set `--eval-episodes` to the desired number. Its `collect_failure_cases.py` helper writes observations, images, and actions to `failed_eval_demos`. Restore the VLM client assignment before evaluating CoC. Failure saving is disabled by default.

### 2. Generate and check CoT annotations
Use the simulator/policy environment for this step: 
```bash
conda activate bridgevla
cd "$COC_ROOT"
export PYTHONPATH="$COC_ROOT:${PYTHONPATH:-}"

TASK=move_hanger
xvfb-run -a python data_preparation/build_dataset.py \
  --task "$TASK" --episodes 100 --mode success \
  --output_dir /data/colosseum_dataset/expert

xvfb-run -a python data_preparation/build_dataset.py \
  --task "$TASK" --mode fail \
  --output_dir /data/colosseum_dataset/failure

python data_preparation/check_task_abc.py \
  "/data/colosseum_dataset/expert/$TASK.json" --tolerance 0.02
python data_preparation/check_task_abc.py \
  "/data/colosseum_dataset/failure/$TASK.json" --tolerance 0.02
```

### 3. Balance correction samples
```bash
python data_preparation/build_balanced_dataset.py \
  --expert_dir /data/colosseum_dataset/expert \
  --fail_dir /data/colosseum_dataset/failure \
  --output_path /data/colosseum_dataset/mixed_sft.json \
  --target_ratio 0.20
```
This script oversamples correction-needed frames and checks image-token alignment. Keep only the final JSON file, and preserve the image paths recorded in the annotations.

## 🚀 Train CoC
Set `model.path` in `configs/sft_config.yaml` to your MomaGraph-R1 directory. The actual training input is selected by `--data-path`.
```bash
conda activate cot_sft
cd "$COC_ROOT"
export PYTHONPATH="$COC_ROOT:${PYTHONPATH:-}"
export WANDB_MODE=offline  # Optional: keep logging local

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 --master_port=29516 \
  scripts/train_sft_cot.py \
  --config configs/sft_config.yaml \
  --data-path /data/colosseum_dataset/mixed_sft.json \
  --output-dir /data/colosseum_dataset/checkpoints/sft/coc \
  --run-name coc-sft
```

## ⚖️ Evaluation: Online policy steering
Start one VLM server in the VLM environment after configuring its base-model path:

```bash
conda activate cot_sft
cd "$COC_ROOT"
CUDA_VISIBLE_DEVICES=0 python vlm_corrector_server.py \
  --ckpt /data/colosseum_dataset/checkpoints/sft/coc/checkpoint-epoch-1 \
  --gpu cuda:0 --port 5050 --ablation none
```

The server loads raw SFT weights (`pytorch_model.bin` or `model.safetensors`) or a PEFT adapter. Its model and tokenizer must match the base model used for training. Check the weight-loading diagnostics before rollout.

In another terminal, activate the policy environment and evaluate one task. Update the worker's local paths first:

```bash
conda activate bridgevla
cd "$COC_ROOT"
export PYTHONPATH="$COC_ROOT:${PYTHONPATH:-}"
VLM_PORT=5050 bash policy/BridgeVLA/finetune/Colosseum/eval_worker.sh \
  trial1 80 /path/to/BridgeVLA/COLOSSEUM 1 "move_hanger"
```

The arguments are `<trial> <policy_epoch> <policy_checkpoint_directory> <physical_gpu_id> <task_list>`. Here the VLM uses GPU 0 and the policy worker uses GPU 1. The worker iterates variation IDs `0..14`, evaluates 25 episodes for each available task variation, and writes results below the policy checkpoint directory. It can move legacy `episodes/` folders into `variation0/`; provide an already organized evaluation dataset when possible.

Use a separate VLM process and port for each concurrent rollout worker: the server stores a single episode history. The client resets that history at episode boundaries. If a server request fails, the current client logs an error and falls back to the base proposal; verify connectivity so a fallback run is not mistaken for CoC evaluation.

## 🙏 Acknowledgements
This project builds on [BridgeVLA](https://github.com/BridgeVLA/BridgeVLA), [MomaGraph](https://github.com/HybridRobotics/MomaGraph), and [COLOSSEUM](https://github.com/robot-colosseum/robot-colosseum). We thank their authors for making these resources available. 

## 📝 Citation
If you find this work useful in your research, please cite our paper:
```bibtex
@inproceedings{zhang2026coc,
  title={Chain-of-Correction: Progress-Aware Policy Steering via Anchor-Grounded Predictive Reasoning},
  author={Xuening Zhang and Xiang Deng and Jiayi Lin and Qi Lv and Xingbo Liu and Weili Guan},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026}
}
```

## ✉️ Contact
If you have any questions or find any bugs, please feel free to open an issue or contact [Xuening Zhang](mailto:yukiZhang0527@outlook.com).
