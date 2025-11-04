# 模型文件组织说明

## 文件命名规则

为避免不同模型类型之间的文件冲突，系统自动为模型文件添加后缀：

### Multi-Task 模型（推荐）
```
models/
├── tfidf_vectorizer_multitask.json    # TF-IDF 词汇表
└── mlp_weights_multitask.pt           # 神经网络权重（4个二分类头）
```

**特点**：
- 4 个独立的二分类器：E/I, S/N, T/F, J/P
- 无需 class_mapping 文件（固定映射）
- 更符合 MBTI 理论

### Single-Task 模型
```
models/
├── tfidf_vectorizer_single.json       # TF-IDF 词汇表
├── mlp_weights_single.pt              # 神经网络权重（16类输出）
└── class_mapping_single.json          # 16 类标签映射
```

**特点**：
- 直接 16-way 分类
- 需要 class_mapping 文件存储标签映射
- 实现简单

## 训练不同模型

### 1. 训练 Multi-Task 模型

编辑 `config.toml`:
```toml
[model]
model_type = "multitask"
```

运行：
```bash
./target/release/psycial hybrid train
```

保存到：
- `models/tfidf_vectorizer_multitask.json`
- `models/mlp_weights_multitask.pt`

### 2. 训练 Single-Task 模型

编辑 `config.toml`:
```toml
[model]
model_type = "single"
```

运行：
```bash
./target/release/psycial hybrid train
```

保存到：
- `models/tfidf_vectorizer_single.json`
- `models/mlp_weights_single.pt`
- `models/class_mapping_single.json`

## 并行训练

可以训练多个模型版本进行对比：

```bash
# 1. 训练 multi-task 模型
# config.toml: model_type = "multitask"
./target/release/psycial hybrid train

# 2. 修改配置
# config.toml: model_type = "single"

# 3. 训练 single-task 模型
./target/release/psycial hybrid train

# 两个模型不会冲突！
```

## 模型对比

训练完两种模型后，`models/` 目录结构：

```
models/
├── tfidf_vectorizer_multitask.json
├── mlp_weights_multitask.pt
├── tfidf_vectorizer_single.json
├── mlp_weights_single.pt
└── class_mapping_single.json
```

可以对比两种方法的效果：

| 模型类型 | 文件数量 | 模型大小 | 预期准确率 |
|---------|---------|---------|-----------|
| Multi-task | 2 个 | ~40MB | **55-60%** ⬆️ |
| Single-task | 3 个 | ~40MB | 49.16% |

## 预测时选择模型

目前预测功能需要手动指定模型文件（未来可以扩展）。

建议在训练时记录最佳模型的类型，用于后续预测。

## 清理模型

```bash
# 删除所有 multi-task 模型
rm models/*_multitask.*

# 删除所有 single-task 模型
rm models/*_single.*

# 删除所有模型
rm -rf models/
```

## 最佳实践

1. **实验阶段**：两种模型都训练，对比效果
2. **生产环境**：选择效果最好的模型类型
3. **版本管理**：
   ```bash
   # 保存实验结果
   cp models/mlp_weights_multitask.pt models/mlp_weights_multitask_v1_acc55.pt
   ```
4. **Git 忽略**：
   ```gitignore
   # .gitignore
   models/*.pt
   models/*.json
   ```

---

🎯 **推荐**：优先使用 **multi-task** 模型，理论和实践都证明其效果更好！

