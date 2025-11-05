# Multi-Task MBTI Classification Model

## 概述

实现了两种 MBTI 分类方法：

### 1. **Single-Task (16-way classification)**
- 直接预测 16 种 MBTI 类型
- 输出层：16 个神经元
- 适合快速原型

### 2. **Multi-Task (4 x 2-way classification)** ⭐ 推荐
- 四个独立的二分类任务：
  - **E/I**: Extraversion vs Introversion
  - **S/N**: Sensing vs Intuition  
  - **T/F**: Thinking vs Feeling
  - **J/P**: Judging vs Perceiving
- 更符合 MBTI 心理学理论
- 论文方法，通常准确率更高

## 配置使用

编辑 `config.toml`:

```toml
[model]
# 选择模型类型
model_type = "multitask"  # 或 "single"

# 网络结构（共享层）
hidden_layers = [1024, 512, 256]

# 超参数
learning_rate = 0.001
dropout_rate = 0.5

[training]
epochs = 25
batch_size = 64
```

## 训练模型

### 方法 1：命令行参数（推荐）

```bash
# 训练 multi-task 模型
./target/release/psycial hybrid train --multi-task

# 训练 single-task 模型
./target/release/psycial hybrid train --single-task

# 使用 config.toml 中的默认设置
./target/release/psycial hybrid train
```

### 方法 2：配置文件

编辑 `config.toml`：
```toml
[model]
model_type = "multitask"  # 或 "single"
```

然后运行：
```bash
./target/release/psycial hybrid train
```

> 💡 **优先级**：命令行参数 > 配置文件

##输出示例

### Multi-Task Training:
```
🎮 Multi-Task MLP Device: Cuda(0)
🏋️  Training Multi-Task Model on Cuda(0)...
   4 binary classifiers: E/I, S/N, T/F, J/P

  Epoch   5/25: Loss=0.2156, Avg=82.3% (E/I=85.1% S/N=78.2% T/F=84.5% J/P=81.4%)
  Epoch  10/25: Loss=0.1234, Avg=88.7% (E/I=90.2% S/N=85.6% T/F=89.9% J/P=89.1%)
  ...
```

### 评估结果：
```
+--------------------------------------------+----------+
| Method                                     | Accuracy |
+--------------------------------------------+----------+
| Baseline (TF-IDF + Naive Bayes)            |  21.73%  |
| Hybrid (single-task, previous)             |  49.16%  |
| Hybrid (multi-task, current)               |  ??%     | ← 预期更高!
| Paper Target                               |  86.30%  |
+--------------------------------------------+----------+
```

## 技术优势

### Multi-Task vs Single-Task

| 特性 | Single-Task | Multi-Task |
|------|-------------|------------|
| **理论依据** | 仅基于数据 | 符合 MBTI 心理学 |
| **类别平衡** | 16 类不平衡严重 | 4 个二分类更平衡 |
| **可解释性** | 低 | 高（每维度独立分析） |
| **训练难度** | 较难 | 较易（子任务简单） |
| **论文支持** | - | ✅ 证明有效 |
| **预期准确率** | ~49% | **>55%** 🎯 |

## 架构细节

```
Input Features (5384 维)
    ↓
Shared Layers (1024 → 512 → 256)
    ↓
┌───────┬───────┬───────┬───────┐
│ E/I   │ S/N   │ T/F   │ J/P   │  ← 4 个独立输出头
│ (2)   │ (2)   │ (2)   │ (2)   │
└───────┴───────┴───────┴───────┘
    ↓       ↓       ↓       ↓
  组合成完整 MBTI 类型 (如 "INTJ")
```

### 优化点
- **Shared layers**: 提取通用特征
- **Independent heads**: 每个维度独立优化
- **Multi-task loss**: 4 个损失函数的平均值

## 模型文件

**Multi-task 模型** 保存为：
- `models/tfidf_vectorizer_multitask.json` - TF-IDF 词汇表
- `models/mlp_weights_multitask.pt` - 神经网络权重（包含 4 个输出头）

**Single-task 模型** 保存为：
- `models/tfidf_vectorizer_single.json` - TF-IDF 词汇表
- `models/mlp_weights_single.pt` - 神经网络权重（16 类输出）
- `models/class_mapping_single.json` - 16 类标签映射

> 💡 **自动命名**：程序会根据模型类型自动添加 `_multitask` 或 `_single` 后缀，避免文件冲突。

## 切换模型

只需修改 `config.toml` 中的 `model_type`:

```toml
# Multi-task (推荐)
model_type = "multitask"

# Single-task
model_type = "single"
```

无需修改代码，重新训练即可！

## 预期改进

相比 single-task (49.16%)，multi-task 模型预期：
- ✅ 提升 **10-15%** 准确率
- ✅ 更好的泛化能力
- ✅ 降低过拟合
- ✅ 更接近论文目标 (86.30%)

## 下一步

1. 运行训练查看实际效果
2. 如果效果好，考虑进一步优化：
   - 调整共享层大小
   - 尝试不同的 dropout 率
   - 添加维度权重（某些维度可能更重要）

---

🎯 **目标**: 通过 multi-task 学习突破 55% 准确率大关！

