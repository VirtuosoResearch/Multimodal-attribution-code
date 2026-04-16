# pid_estimator

从 `/data/zhenshuo/multimodal/PID`、`/data/zhenshuo/multimodal/LSMI_Estimator` 和当前仓库的 surrogate 设计中整理出来的可调用模块，当前包含四部分：

- `ce_alignment.py`: CE alignment PID estimator，支持直接传入 `x1/x2/labels` 或自定义 train/test dataset。
- `synthetic_data.py`: synthetic data 生成、pickle 数据读写、二值逻辑 toy case 和解析基线。
- `lsmi.py`: LSMI estimator，保留原仓库的分类器 + 熵估计器结构，但改成了直接可 import 的函数接口。
- `surrogate.py`: 基于 `new_algo.md` 的 subset-calibrated surrogate estimator，先学 `q1/q2/q12`，再用 calibration subsets 拟合 redundancy head。

最短调用示例：

```python
from pid_estimator import (
    CEAlignmentConfig,
    estimate_pid_with_ce_alignment,
    make_binary_logic_dataset,
)

x1, x2, y, dataset = make_binary_logic_dataset("xor", repeat=256)
config = CEAlignmentConfig(batch_size=64, discrim_epochs=5, ce_epochs=5)
result = estimate_pid_with_ce_alignment(
    x1,
    x2,
    y,
    num_labels=2,
    train_dataset=dataset,
    test_dataset=dataset,
    config=config,
)

print(result.mean_components)
```

synthetic data 生成示例：

```python
from pid_estimator import SyntheticGenerationConfig, generate_synthetic_dataset

config = SyntheticGenerationConfig(setting="official", rho=0.5, noise_degree=0.5, num_data=4096, seed=0)
data = generate_synthetic_dataset(config)
print(data["train"]["0"].shape, data["train"]["label"].shape)
```

LSMI 调用示例：

```python
from pid_estimator import LSMIConfig, estimate_lsmi, make_binary_logic_dataset

x1, x2, y, dataset = make_binary_logic_dataset("xor", repeat=256)
config = LSMIConfig(batch_size=64, discriminator_epochs=5, entropy_epochs=5)
result = estimate_lsmi(
    x1,
    x2,
    y,
    num_labels=2,
    train_dataset=dataset,
    test_dataset=dataset,
    config=config,
)

print(result.mean_components)
```

Surrogate estimator 调用示例：

```python
from pid_estimator import SurrogatePIDConfig, estimate_surrogate_pid, make_binary_logic_dataset

x1, x2, y, dataset = make_binary_logic_dataset("xor", repeat=256)
config = SurrogatePIDConfig(
    batch_size=64,
    predictor_epochs=5,
    head_epochs=100,
    calibration_blocks=8,
    calibration_subsets=10,
)
result = estimate_surrogate_pid(
    x1,
    x2,
    y,
    num_labels=2,
    train_dataset=dataset,
    test_dataset=dataset,
    config=config,
)

print(result.mean_components)
print(result.query_engine.query(torch.ones(len(result.block_indices))))
```
