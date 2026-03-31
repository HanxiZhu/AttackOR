# Attack_OR: 对抗性优化推理训练项目

本项目整合了针对运筹学问题的对抗攻击生成、SFT训练和GRPO强化学习训练的完整流程。

## 项目结构

```
Attack_OR/
├── attacks/                           # 攻击生成模块
│   ├── attack.py                      # 攻击生成器主入口
│   ├── attack_with_attribute_library.py  # Attribute攻击
│   ├── attack_with_phantom_library.py    # Phantom攻击
│   ├── attack_with_arithmetic.py         # Arithmetic攻击
│   ├── attack_with_unit_conversion.py    # Unit Conversion攻击
│   ├── build_attribute_library.py        # 构建属性库脚本
│   └── attribute_library.json            # 属性库
├── sft/                               # SFT训练模块
│   ├── finetune.py                    # 主训练脚本
│   ├── data.py                        # 数据处理
│   ├── arguments.py                   # 参数定义
│   └── configs/                       # DeepSpeed配置
├── grpo/                              # GRPO训练模块
│   ├── scripts/                       # 训练脚本
│   │   ├── run_attack_or.sh
│   │   ├── run_attack_or_4k.sh
│   │   ├── run_attack_or_ablation.sh
│   │   └── run_attack_or_ablation_4k.sh
│   ├── config/                        # 配置文件
│   │   └── attack_or_grpo.yaml
│   └── rewards/                       # 奖励函数
│       └── attack_or.py
├── eval/                              # 评估模块
│   ├── generate.py                    # 生成求解代码
│   ├── execute.py                     # 执行评估
│   ├── generate_attack.py             # 攻击评估生成
│   └── execute_attack.py              # 攻击评估执行
├── data/                              # GRPO训练数据(parquet格式)
│   └── *.parquet
├── datasets_local/                    # 原始数据集(jsonl格式)
│   └── *.jsonl
├── data_preprocess/                   # 数据预处理脚本
│   ├── convert_to_verl.py             # 转换为verl格式
│   ├── convert_to_eval.py             # 转换为评测格式
│   ├── merge_datasets.py              # 合并数据集
│   ├── split_dataset.py               # 划分数据集
│   ├── shuffle_data.py                # 打乱数据
│   └── inspect_data.py                # 检查数据
├── scripts/                           # 辅助脚本
│   ├── download_datasets.py           # 下载原始数据集
│   ├── merge_adversarial_datasets.py  # 合并对抗数据集
│   └── filter_attack_types.py         # 过滤攻击类型(消融实验)
├── train.sh                           # SFT训练启动脚本
├── search.sh                          # 超参数搜索脚本
├── evaluate_attack.sh                 # 攻击评估脚本
├── verl/                              # verl核心模块
│   ├── trainer/
│   ├── workers/
│   ├── utils/
│   └── ...
├── pyproject.toml                     # 项目配置
├── requirements.txt                   # Python依赖
└── README.md                          # 项目说明
```

## 四种攻击类型

1. **Attribute Attack**: 基于属性库的干扰属性注入
2. **Phantom Attack**: 虚幻择优选项攻击
3. **Arithmetic Attack**: 算术等价替换攻击
4. **Unit Conversion Attack**: 单位换算攻击

## 完整工作流程

### 步骤1: 下载原始数据集

```bash
python scripts/download_datasets.py --dataset all --output_dir ./datasets_local
```

### 步骤2: 构建属性库(首次运行)

```bash
cd attacks
python build_attribute_library.py --model /path/to/model --output ../attacks/attribute_library.json
```

### 步骤3: 生成攻击数据

```bash
cd attacks
python attack.py --input ../datasets_local/IndustryOR_test.jsonl --output ../datasets_local/IndustryOR_test_adversarial.jsonl
```

### 步骤4: 合并攻击数据集

```bash
python scripts/merge_adversarial_datasets.py --data-dir ./datasets_local
```

### 步骤5: 转换为verl训练格式

```bash
cd data_preprocess
python convert_to_verl.py --input ../datasets_local/merged_attack.jsonl --output ../data/merged_train.parquet
```

### 步骤6: SFT训练

```bash
# 单次训练
bash train.sh

# 或超参数搜索
bash search.sh
```

### 步骤7: GRPO训练

```bash
cd grpo/scripts

# 完整训练
bash run_attack_or.sh vllm /path/to/actor_model /path/to/llm_model

# 4K context训练
bash run_attack_or_4k.sh vllm /path/to/actor_model /path/to/llm_model

# 消融实验
ATTACK_TYPES="baseline,arithmetic" bash run_attack_or_ablation.sh
```

### 步骤8: 评估

```bash
bash evaluate_attack.sh ./datasets_local/test_adversarial.jsonl ./results /path/to/model
```

## 奖励函数说明

GRPO训练使用自定义奖励函数 (`grpo/rewards/attack_or.py`)，采用阶梯式奖励机制：

- **格式分 (+0.2)**: 生成有效的Python代码
- **运行分 (+0.3)**: 代码成功执行并返回结果
- **准确分 (+1.0)**: 结果正确（5%相对误差容忍）
- **防御逻辑分 (+1.0)**: 正确识别并处理攻击中的干扰信息
- **长度惩罚 (-0~-3)**: 过长输出会被惩罚

## 消融实验

支持按攻击类型过滤数据进行消融实验：

```bash
# 仅使用baseline和arithmetic攻击类型
ATTACK_TYPES="baseline,arithmetic" bash run_attack_or_ablation.sh

# 使用指定攻击类型组合
ATTACK_TYPES="baseline,unit_conversion,llm_attribute" bash run_attack_or_ablation.sh
```

可用的攻击类型：
- `baseline`: 原始数据
- `arithmetic`: 算术攻击
- `unit_conversion`: 单位换算攻击
- `llm_attribute`: Attribute攻击
- `llm_phantom`: Phantom攻击

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 依赖项目

- [ORLM](https://github.com/orlm/ORLM): 运筹学大语言模型
- [verl](https://github.com/volcengine/verl): 强化学习训练框架

## License

Apache 2.0
# AttackOR
