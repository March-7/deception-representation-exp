# IFLOW 上下文文档

## 项目概述

这是一个研究项目，配套论文 **"When Truthful Representations Flip Under Deceptive Instructions?"**（[arXiv链接](https://www.arxiv.org/pdf/2507.22149)）。该项目研究欺骗性指令如何重塑大语言模型（LLM）的内部表征，相比于真实/中性提示词。研究包括输出层面的评估和表征层面的分析（线性探针和稀疏自编码器）。

### 核心研究内容

- **三种提示词类型评估**：在事实验证任务上评估 LLMs，使用三种提示词人设：Truthful（真实）、Deceptive（欺骗）、Neutral（中性）
- **线性可解码性**：证明模型的预期 True/False 输出在所有指令类型下都可以从隐藏状态线性解码
- **表征偏移量化**：使用稀疏自编码器（SAEs）量化欺骗引起的表征偏移（L2、余弦相似度、重叠度），集中在早期到中期层；真实和中性提示词高度对齐
- **诚实子空间识别**：识别对欺骗敏感的 SAE 特征，定义紧凑的"诚实子空间"

### 技术栈

- **语言**: Python 3.10
- **核心框架**: PyTorch, Transformers (Hugging Face), sae-lens
- **数据处理**: pandas, numpy
- **机器学习**: scikit-learn (LogisticRegression, 探针训练)
- **可视化**: matplotlib, seaborn, plotly
- **降维**: umap-learn, TSNE
- **模型支持**: Llama3/3.1/2, Gemma/Gemma2, Mistral, Qwen2.5, DeepSeek

## 项目结构

```
deception-representation-exp/
├── config.ini                    # 模型权重路径配置
├── requirements.txt              # pip 依赖
├── environment.yml               # conda 环境配置
├── datasets/                     # 数据集目录
│   ├── The Geometry of Truth/    # 来自 "The Geometry of Truth" 的数据集
│   └── The Internal State.../    # 来自 "The Internal State..." 的数据集
├── scripts/                      # 主要实验脚本
│   ├── extract_activations.py    # 提取模型层激活
│   ├── run_probing_pipeline.py   # 训练和评估探针
│   ├── vis_probe_results.py      # 可视化探针准确率
│   ├── analyze_feature_shift_sae.py  # SAE 特征偏移分析
│   ├── vis_feature_shift.py      # 可视化特征偏移
│   ├── eval_instruction_following.py # 输出层面准确率评估
│   └── summarize_accuracy_by_category.py # 按类别汇总准确率
├── src/                          # 核心工具模块
│   ├── probes.py                 # 探针类定义 (TTPD, LRProbe)
│   ├── sae_utils.py              # SAE 工具函数
│   └── utils.py                  # 数据管理和统计工具
├── experimental_outputs/         # 实验结果输出目录（运行时生成）
└── llm_weights/                  # 模型权重缓存目录（运行时生成）
```

## 环境安装

### 方法 A：使用 pip（推荐）

```bash
pip install -U pip
pip install -r requirements.txt
```

### 方法 B：使用 conda

```bash
conda env create -f environment.yml
conda activate truth_flip
```

### Hugging Face 认证

对于需要门控访问的模型（如 Llama 系列）：

```bash
export HF_TOKEN="<your_huggingface_token>"
python -c "from huggingface_hub import login; import os; login(token=os.getenv('HF_TOKEN'))"
```

## 快速开始：最小化复现流程

以下流程使用 Gemma-2 2B chat 模型进行演示，资源需求较低。

### 1. 提取残差流激活

为三种提示词类型提取层激活：

```bash
# Truthful 提示词
python scripts/extract_activations.py \
  --model_family Gemma2 \
  --model_size 2B \
  --model_type chat \
  --prompt_type truthful \
  --layers -1 \
  --datasets all \
  --device cuda:0

# Neutral 提示词
python scripts/extract_activations.py \
  --model_family Gemma2 \
  --model_size 2B \
  --model_type chat \
  --prompt_type neutral \
  --layers -1 \
  --datasets all \
  --device cuda:0

# Deceptive 提示词
python scripts/extract_activations.py \
  --model_family Gemma2 \
  --model_size 2B \
  --model_type chat \
  --prompt_type deceptive \
  --layers -1 \
  --datasets all \
  --device cuda:0
```

**参数说明**：
- `--layers -1`: 提取所有层
- `--datasets all`: 使用所有可用数据集
- `--device cuda:0`: 使用第一块 GPU

### 2. 运行探针流程

训练线性探针（LR）和 TTPD 探针：

```bash
python scripts/run_probing_pipeline.py
```

此脚本会：
- 对所有层和提示词类型训练探针
- 执行交叉验证
- 保存探针准确率结果

### 3. 可视化探针结果

生成准确率曲线和表格：

```bash
python scripts/vis_probe_results.py \
  --model_family Gemma2 \
  --model_size 9B \
  --model_type chat \
  --save_dir experimental_outputs/probing_and_visualization/accuracy_figures
```

### 4. SAE 特征偏移分析（可选）

需要 `sae-lens` 检查点：

```bash
python scripts/analyze_feature_shift_sae.py
python scripts/vis_feature_shift.py
```

## 主要脚本说明

### 激活提取 (`extract_activations.py`)

从指定模型层提取残差流激活，支持：
- 多种模型家族（Llama, Gemma, Mistral 等）
- 三种提示词类型（truthful, neutral, deceptive）
- 自定义层选择
- 批量数据集处理

**输出位置**: `acts/acts_{prompt_type}_prompt/{model_family}/{model_size}/{model_type}/{dataset}/`

### 探针训练流程 (`run_probing_pipeline.py`)

实现完整的探针训练和评估流程：
- **探针类型**: 
  - `TTPD`: Truth and Polarity Direction Probe（真实性和极性方向探针）
  - `LRProbe`: Logistic Regression Probe（逻辑回归探针）
- **交叉验证**: 20 次迭代，确保结果稳定性
- **多层分析**: 逐层训练和评估

**输出位置**: `experimental_outputs/probing_and_visualization/{prompt_type}/{model_family}/{model_size}/{model_type}/`

### 探针可视化 (`vis_probe_results.py`)

生成：
- 准确率-层数曲线图（带误差条）
- Markdown 格式的准确率表格
- 支持多种提示词类型对比

### SAE 分析 (`analyze_feature_shift_sae.py`)

使用稀疏自编码器分析特征偏移：
- **度量指标**: L1, L2, 余弦相似度, 特征重叠度
- **提示词对比**: truthful vs deceptive, truthful vs neutral, neutral vs deceptive
- **多层并行分析**

**关键配置**（脚本内修改）：
```python
MODEL_RELEASE = "gemma-scope-2b-pt-res-canonical"
LAYERS_TO_ANALYZE = list(range(32))
DATASETS = ["common_claim_true_false", "counterfact_true_false", ...]
```

### 指令遵循评估 (`eval_instruction_following.py`)

评估模型在不同提示词下的输出层面准确率，保存 CSV 结果。

## 数据集格式

所有数据集为 CSV 格式，包含两列：
- `statement` (string): 陈述文本
- `label` (int): 标签，1=真，0=假

**示例数据集**：
- `cities.csv`: 城市相关事实
- `common_claim_true_false.csv`: 常识性声明
- `counterfact_true_false.csv`: 反事实声明
- `sp_en_trans.csv`: 西班牙语-英语翻译
- `animal_class.csv`, `element_symb.csv`, `facts.csv` 等

## 配置文件 (`config.ini`)

配置各模型家族的权重路径。当前配置使用本地路径 `/data1/ckx/hf-checkpoints`：

```ini
[Gemma2]
weights_directory = /data1/ckx/hf-checkpoints/google/gemma-2-2b-it
2B_chat_subdir = gemma2_2b_it_hf
9B_chat_subdir = gemma2_9b_it_hf

[Llama3.1]
weights_directory = /data1/ckx/hf-checkpoints/meta-llama/Llama-3.1-8B-Instruct
8B_chat_subdir = llama3.1_8b_chat_hf

[Llama3.3]
weights_directory = /data1/ckx/hf-checkpoints/meta-llama/Llama-3.3-70B-Instruct
70B_chat_subdir = llama3.3_70b_chat_hf

[Qwen3]
weights_directory = /data1/ckx/hf-checkpoints/Qwen/Qwen3-8B
8B_subdir = qwen3_8b
14B_subdir = qwen3_14b
32B_subdir = qwen3_32b
```

**当前可用模型**：
- **Gemma2**: 2B-it, 9B-it
- **Llama3.1**: 8B-Instruct
- **Llama3.3**: 70B-Instruct
- **Qwen3**: 8B, 14B, 32B

**使用说明**：
- 所有模型权重存储在 `/data1/ckx/hf-checkpoints/` 目录
- 可以使用 Hugging Face 模型标识符（如 `google/gemma-2-2b-it`）自动下载
- 或使用绝对路径指向本地模型文件
- 根据需要修改 `weights_directory` 字段指向你的模型存储位置

## 核心模块 (`src/`)

### `probes.py`

定义两种探针类：

**TTPD (Truth and Polarity Direction Probe)**：
- 学习真实性方向 (`t_g`) 和极性方向 (`t_p`)
- 使用 OLS 解析解学习方向
- 将激活投影到 2D 空间后用 LR 分类

**LRProbe (Logistic Regression Probe)**：
- 标准逻辑回归探针
- 直接在原始激活上训练

### `sae_utils.py`

SAE（稀疏自编码器）相关工具函数，用于：
- 加载预训练 SAE 模型
- 编码/解码激活
- 特征提取

### `utils.py`

数据管理和统计工具：
- `DataManager`: 管理数据集加载和批处理
- `collect_training_data`: 收集训练数据并计算中心化激活
- `compute_statistics`: 计算统计指标
- `compute_average_accuracies`: 计算平均准确率

## 输出目录结构

运行实验后，输出将组织在 `experimental_outputs/` 下：

```
experimental_outputs/
├── probing_and_visualization/
│   ├── truthful/
│   │   └── Gemma2/
│   │       └── 2B/
│   │           └── chat/
│   │               ├── probe_results_*.pkl
│   │               └── accuracy_curves.png
│   ├── deceptive/
│   └── neutral/
└── feature_shift_results_*/
    ├── layer_*.npz  # SAE 特征偏移结果
    └── summary_*.csv
```

## 开发约定

### 代码风格

- **类型提示**: 使用 Python 类型提示（如 `def func(x: int) -> str:`）
- **命名**: 
  - 变量/函数: `snake_case`
  - 类: `PascalCase`
  - 常量: `UPPER_CASE`
- **导入**: 按标准库、第三方库、本地模块顺序组织

### 实验流程

1. **激活提取**: 首先运行 `extract_activations.py` 为所有提示词类型提取激活
2. **探针训练**: 运行 `run_probing_pipeline.py` 训练探针
3. **结果可视化**: 使用 `vis_probe_results.py` 生成图表
4. **SAE 分析**: （可选）运行 SAE 相关脚本进行深度分析

### 添加新模型

1. 在 `config.ini` 中添加新模型配置
2. 确保模型与 Hugging Face `transformers` 兼容
3. 根据需要调整 `extract_activations.py` 中的 `load_model` 函数

### 添加新数据集

1. 准备 CSV 文件，包含 `statement` 和 `label` 列
2. 将文件放入 `datasets/` 目录
3. 在脚本中的数据集列表中添加文件名（不含 `.csv` 扩展名）

## 常见问题

### GPU 内存不足

- 使用较小的模型（如 Gemma-2 2B）
- 减少批处理大小
- 使用 `torch.cuda.empty_cache()` 清理缓存
- 设置环境变量: `export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`

### 模型加载失败

- 检查 `config.ini` 中的路径是否正确
- 确保已完成 Hugging Face 认证（对于门控模型）
- 验证磁盘空间是否充足（模型权重可能很大）

### 数据集未找到

- 确认 CSV 文件在 `datasets/` 目录中
- 检查文件名拼写（区分大小写）
- 验证 CSV 格式（必须包含 `statement` 和 `label` 列）

## 引用

如果使用此代码或数据集，请引用论文。详见 `CITATION.cff`。

## 许可证

代码使用 MIT 许可证发布（见 `LICENSE`）。模型和数据集许可证归其各自所有者所有。

## 相关论文

- **主论文**: "When Truthful Representations Flip Under Deceptive Instructions?" ([arXiv:2507.22149](https://www.arxiv.org/pdf/2507.22149))
- **相关工作**: 
  - "The Geometry of Truth" (数据集来源)
  - "The Internal State of an LLM Knows When It's Lying" (数据集来源)

---

**注意**: 此项目用于研究目的。在使用模型和数据集时，请遵守相应的使用条款和许可证。
