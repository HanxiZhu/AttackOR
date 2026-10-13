# AttackOR

**Are LLMs Robust Enough for Operations Research? An Adversarial Attack and Defense Study via AttackOR**

Hanxi Zhu, Jingru Yu, Yan Teng, Jian Wang, Yi Yu · **COLM 2026 (The Third Conference on Language Modeling)**

[Paper PDF](https://hanxizhu.github.io/attackor-project-page/assets/attack_or_colm2026.pdf) · [OpenReview](https://openreview.net/forum?id=jT4fTtBw3F) · [Project page](https://hanxizhu.github.io/attackor-project-page/) · [中文 README](README.md)

Conference poster presentation: [Poster Session 1, Imperial Ballroom, #11](https://colm.eventhosts.cc/Conferences/2026/AcceptedPapers).

## Citation

```bibtex
@inproceedings{zhu2026attackor,
  title={Are {LLMs} Robust Enough for Operations Research? An Adversarial Attack and Defense Study via {AttackOR}},
  author={Zhu, Hanxi and Yu, Jingru and Teng, Yan and Wang, Jian and Yu, Yi},
  booktitle={The Third Conference on Language Modeling},
  year={2026},
  url={https://openreview.net/forum?id=jT4fTtBw3F}
}
```

## Overview

This project provides an end-to-end pipeline for **adversarial attack generation** on Operations Research (OR) / optimization problems, followed by **SFT fine-tuning** and **GRPO reinforcement learning** training.

## Project Structure

```
Attack_OR/
├── attacks/                           # Attack generation
│   ├── attack.py                      # Main entry for attack generation
│   ├── attack_with_attribute_library.py  # Attribute attack
│   ├── attack_with_phantom_library.py    # Phantom attack
│   ├── attack_with_arithmetic.py         # Arithmetic attack
│   ├── attack_with_unit_conversion.py    # Unit conversion attack
│   ├── build_attribute_library.py        # Build attribute library
│   └── attribute_library.json            # Attribute library
├── sft/                               # SFT training
│   ├── finetune.py                    # Main training script
│   ├── data.py                        # Data processing
│   ├── arguments.py                   # Argument definitions
│   └── configs/                       # DeepSpeed configs
├── grpo/                              # GRPO training
│   ├── scripts/                       # Training scripts
│   │   ├── run_attack_or.sh
│   │   ├── run_attack_or_4k.sh
│   │   ├── run_attack_or_ablation.sh
│   │   └── run_attack_or_ablation_4k.sh
│   ├── config/                        # Config files
│   │   └── attack_or_grpo.yaml
│   └── rewards/                       # Reward functions
│       └── attack_or.py
├── eval/                              # Evaluation
│   ├── generate.py                    # Generate solver code
│   ├── execute.py                     # Execute evaluation
│   ├── generate_attack.py             # Generate attack evaluation
│   └── execute_attack.py              # Execute attack evaluation
├── data/                              # GRPO training data (parquet)
│   └── *.parquet
├── datasets_local/                    # Raw datasets (jsonl)
│   └── *.jsonl
├── data_preprocess/                   # Preprocessing scripts
│   ├── convert_to_verl.py             # Convert to verl format
│   ├── convert_to_eval.py             # Convert to eval format
│   ├── merge_datasets.py              # Merge datasets
│   ├── split_dataset.py               # Split dataset
│   ├── shuffle_data.py                # Shuffle data
│   └── inspect_data.py                # Inspect data
├── scripts/                           # Helper scripts
│   ├── download_datasets.py           # Download raw datasets
│   ├── merge_adversarial_datasets.py  # Merge adversarial datasets
│   └── filter_attack_types.py         # Filter attack types (ablation)
├── train.sh                           # SFT launcher
├── search.sh                          # Hyperparameter search launcher
├── evaluate_attack.sh                 # Attack evaluation launcher
├── verl/                              # verl core module
│   ├── trainer/
│   ├── workers/
│   ├── utils/
│   └── ...
├── pyproject.toml                     # Project config
├── requirements.txt                   # Python dependencies
├── README.md                          # Chinese README
└── README_EN.md                       # English README
```

## Four Attack Types

1. **Attribute Attack**: injects distracting attributes based on an attribute library
2. **Phantom Attack**: injects phantom “preferable” options / misleading choices
3. **Arithmetic Attack**: arithmetic-equivalent transformations
4. **Unit Conversion Attack**: unit conversion perturbations

## End-to-End Workflow

### Step 1: Download raw datasets

```bash
python scripts/download_datasets.py --dataset all --output_dir ./datasets_local
```

### Step 2: Build attribute library (first time only)

```bash
cd attacks
python build_attribute_library.py --model /path/to/model --output ../attacks/attribute_library.json
```

### Step 3: Generate adversarial (attack) data

```bash
cd attacks
python attack.py --input ../datasets_local/IndustryOR_test.jsonl --output ../datasets_local/IndustryOR_test_adversarial.jsonl
```

### Step 4: Merge adversarial datasets

```bash
python scripts/merge_adversarial_datasets.py --data-dir ./datasets_local
```

### Step 5: Convert to verl training format

```bash
cd data_preprocess
python convert_to_verl.py --input ../datasets_local/merged_attack.jsonl --output ../data/merged_train.parquet
```

### Step 6: SFT training

```bash
# Single run
bash train.sh

# Or hyperparameter search
bash search.sh
```

### Step 7: GRPO training

```bash
cd grpo/scripts

# Full training
bash run_attack_or.sh vllm /path/to/actor_model /path/to/local_llm_model

# 4K context training
bash run_attack_or_4k.sh vllm /path/to/actor_model /path/to/local_llm_model

# Ablation
ATTACK_TYPES="baseline,arithmetic" bash run_attack_or_ablation.sh
```

### Step 8: Evaluation

```bash
bash evaluate_attack.sh ./datasets_local/test_adversarial.jsonl ./results /path/to/model
```

## Reward Function

GRPO training uses a custom reward function (`grpo/rewards/attack_or.py`) with a step-wise reward scheme:

- **Format (+0.2)**: valid Python code is produced
- **Execution (+0.3)**: the code executes successfully and returns a result
- **Accuracy (+1.0)**: the result is correct (5% relative error tolerance)
- **Defense logic (+1.0)**: correctly identifies and handles injected interference
- **Length penalty (0 to -3)**: overly long outputs are penalized

## Ablation Experiments

You can filter the training data by attack type for ablation studies:

```bash
# Only baseline + arithmetic
ATTACK_TYPES="baseline,arithmetic" bash run_attack_or_ablation.sh

# Custom combinations
ATTACK_TYPES="baseline,unit_conversion,llm_attribute" bash run_attack_or_ablation.sh
```

Available attack types:

- `baseline`: original data
- `arithmetic`: arithmetic attack
- `unit_conversion`: unit conversion attack
- `llm_attribute`: attribute attack
- `llm_phantom`: phantom attack

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Dependencies

- [ORLM](https://github.com/orlm/ORLM): OR language model for optimization
- [verl](https://github.com/volcengine/verl): RL training framework

## License

Apache 2.0
