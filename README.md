# Implicit Turn-Wise Policy Optimization for Proactive User-LLM Interaction

This is the implementation of the paper "Implicit Turn-Wise Policy Optimization for Proactive User-LLM Interaction" by Haoyu Wang, Yuxin Chen, Liang Luo, Buyun Zhang, Ellie Wen, and Pan Li.

## Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
- [3. Code Workflow](#3-code-workflow)
- [4. Citation](#4-citation)
- [5. Acknowledgements](#5-acknowledgements)

---

## 1. Environment Setup

We recommend installing the dependencies with conda.

```bash
conda create -n itpo python=3.10
conda activate itpo
```

### 1.1 Basic Dependencies
To install the dependencies, run the following command:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.0
pip install uv
uv pip install vllm==0.10.2 --torch-backend=cu128
```

### 1.2 Install VeRL

The implementation of ITPO is based on VeRL with version 0.5.0.dev. To install the corresponding version of VeRL, run the following command:

```bash
mkdir ITPO
cd ITPO
git clone https://github.com/verl-project/verl.git
cd verl
git checkout ddd86f52 # The corresponding commit hash of VeRL 0.5.0.dev
cd ..
pip install -e verl/
```

```bash
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder flash_attn==2.8.3+pt2.8.0cu128
pip install ray==2.49.2
pip install litellm==1.79.0
pip install nltk
pip install click==8.2.1
```


### 1.3 Download the ITPO CodeBase

Download the codebase from the repository, and put the itpo folder into the verl/recipe folder of VeRL repo.

For the rest files, simply put them into the corresponding verl/verl/xxx folder.


### 1.4 Issues that may occur

[The Huggingface Hub DNS] When downloading a new model/tokenizer, the xnet version may be incompatible with the current version of transformers. In this case, you can refer to [this issue](https://github.com/huggingface/huggingface_hub/issues/3155) to download the compatible xnet version for fix.

[Click incompatible with Ray] You may download the click version according to [this issue](https://github.com/ray-project/ray/issues/56747) to fix.issue.

[Prebuilt wheels of Flash Attention] If you encounter an error when installing Flash Attention, you can refer to [this issue](https://github.com/Dao-AILab/flash-attention/discussions/1838) to download the prebuilt wheels for fix.


## 2. Data Preparation

```bash
cd ITPO
mkdir data
cd data
mkdir collab
cd ..
cd ..
cd verl
python recipe/itpo/process_data/process_dataset.py --dataset collabllm/collabllm-multiturn-medium-large --local_dir [$HOME PATH of ITPO]/ITPO/data/collab/medium --dataset_type rl
```
One could also organize the [MTMedDialogue](https://github.com/JarvisUSTC/DoctorAgent-RL) following the form above and process the data.

### 2.1 Folder Structure

After data preparation, the folder structure should look like the following:

```text
ITPO
├── data
│   ├── collab
│   │   ├── math
│   │   │   ├── rl_train.parquet
│   │   │   ├── ...
│   │   ...
├── verl
│   ├── recipe/itpo
└── ...
```

## 3. Code Workflow

This section should document the code execution process step by step, including the purpose of each step, the command used, and the expected outputs.

### 3.1 User Simulator / LLM Judge

One should first set up the user simulator and LLM judge via VLLM.

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct --port 8001 --tensor-parallel-size 2 >./vllm_8001.log 2>&1 </dev/null &
```

### 3.2 Training

```bash
bash recipe/itpo/scripts/$Name_of_Script.sh
```

### 3.3. Evaluation

```bash
bash recipe/itpo/scripts/eval/eval.sh
```



## 4. Citation

If you find this work useful for your research, please consider citing the following paper:


```bibtex
@articlewang2026implicit,
  title   = {Implicit Turn-Wise Policy Optimization for Proactive User-LLM Interaction},
  author  = {Wang, Haoyu and Chen, Yuxin and Luo, Liang and Zhang, Buyun and Wen, Ellie and Li, Pan},
  journal = {arXiv preprint arXiv:2603.23550},
  year    = {2026}
}
```


## 5. Acknowledgements

The code implementation is based on [VeRL](https://github.com/verl-project/verl) and [CollabLLM](https://arxiv.org/abs/2502.00640).
