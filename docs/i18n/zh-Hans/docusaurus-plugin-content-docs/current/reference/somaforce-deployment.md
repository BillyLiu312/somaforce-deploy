# SomaForce 部署

本仓库基于 HDMI `sim2real` commit
`d9e1f700667bc75d8d2eeb5ef74bb2a066600612`，复用官方 G1 I/O、ONNX/TensorRT
推理边界、motion backend 和 MuJoCo sim2sim 环路。

## 运行拓扑

```text
HDMI student 或 Sonic nominal -> normalized a_nom[23]
    -> Cross 力觉 residual -> contact gain 与 authority
    -> safety/watchdog -> Unitree G1 或 MuJoCo
```

离线 replay、MuJoCo 和真机使用同一个 `DeploymentStack` 与 action history，只替换
RobotIO 实现。

## 模式

- `hdmi_student_baseline`：只运行 HDMI student。
- `hdmi_student_residual`：主要 zero-shot 力适应路径。
- `sonic_baseline`：不依赖 HDMI object-state 输入的 Sonic。
- `sonic_residual`：Sonic scaffold 加同一个 Cross residual。
- `hdmi_student_residual_shadow` 和 `sonic_residual_shadow`：计算 residual，
  但只发送 nominal action。

## 模型绑定

原生 HDMI student 可以导出为单一 deterministic action graph，也可以导出两个 graph：

```text
adapt_ema(policy[249], object[10]) -> priv_pred[256]
actor_adapt(command[356], policy[249], priv_pred[256]) -> action[23]
```

`HDMIStudentTwoStageNominal` 强制检查上述 shape。Residual 输入为
`wrist_tokens[1,2,16,14]`、`proprio[1,64]`、`a_nom_history[1,3,23]` 和
`previous_a_total[1,23]`，输出 normalized `delta_a[1,23]`。F/T 标定、坐标变换、
接触门控、authority ramp 和物理 action scaling 保持在 ONNX 图外。

Sonic reference 转换必须固定 50 Hz、root-yaw 对齐、future-step 语义和 23 关节映射。
仅有 tensor shape 相同不能证明 residual 兼容。

## Artifact 校验

checkpoint、私有 F/T 标定和传感器 SDK 不放入 Git。以
`configs/tasks/manifest.example.json` 为模板，替换所有 SHA-256 后运行：

```bash
python scripts/verify_artifacts.py artifacts/<task>/manifest.json
```

缺少必需文件、路径穿越和 hash 不匹配都会 fail closed。

## 部署推进

```text
offline replay -> MuJoCo sim2sim -> nominal 真机
    -> residual shadow -> C0 parity -> low-authority pilot
```

MuJoCo 用于验证导出 parity、history/reset、joint mapping、timing、合成 F/T token 和
safety 行为，不能替代 Isaac C1/C2/C3 交互评估。
