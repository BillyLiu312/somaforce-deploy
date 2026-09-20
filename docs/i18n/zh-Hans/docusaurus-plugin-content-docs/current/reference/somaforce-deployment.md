# SomaForce 部署

本仓库基于 HDMI `sim2real` commit
`d9e1f700667bc75d8d2eeb5ef74bb2a066600612`，复用官方 G1 I/O、ONNX/TensorRT
推理边界、motion backend 和 MuJoCo sim2sim 环路。

## 运行拓扑

```text
HDMI student -> normalized a_nom[23]；Sonic G1 nominal -> q_target[29]
    -> Cross 力觉 residual -> contact gain 与 authority
    -> safety/watchdog -> Unitree G1 或 MuJoCo
```

离线 replay、MuJoCo 和真机使用同一个 `DeploymentStack` 与 action history，只替换
RobotIO 实现。

## 模式

- `hdmi_student_baseline`：只运行 HDMI student。
- `hdmi_student_residual`：主要 zero-shot 力适应路径。
- `sonic_baseline`：不依赖 HDMI object-state 输入的 Sonic。
- `sonic_residual`：保留该契约；当前 Sonic G1 导出输出 29 个关节目标，因此
  在明确验证 23 维 adapter 之前只支持 nominal。
- `hdmi_student_residual_shadow` 和 `sonic_residual_shadow`：计算 residual，
  但只发送 nominal action。

## 模型绑定

原生 HDMI student 可以导出为单一 deterministic action graph，也可以导出两个 graph：

```text
adapt_ema(policy[249], command[356], object[10]) -> priv_pred[256]
actor_adapt(command[356], policy[249], priv_pred[256]) -> action[23]
```

`HDMIStudentTwoStageNominal` 强制检查上述 shape。Residual 输入为
`wrist_tokens[1,2,16,14]`、`proprio[1,64]`、`a_nom_history[1,23,3]` 和
`previous_a_total[1,23]`，输出 normalized `delta_a[1,23]`。F/T 标定、坐标变换、
接触门控、authority ramp 和物理 action scaling 保持在 ONNX 图外。

Sonic reference 转换必须固定 50 Hz、root-yaw 对齐和 future-step 语义。共享的
push-door 真机脚本提供 `--param hdmi|sonic`：`sonic` 会把原生 HDMI 的
body/joint motion 转换为 any4hdmi qpos tree，并发布受安全控制器保护的 29 维
G1 proposal；它不会启用 Cross residual。仅凭 shape 相等不能证明 Sonic residual
兼容。

## Push-door-hand 真机入口

真机脚本是 `scripts/run_push_door_hand_hardware.sh`。它沿用 suitcase 脚本的
VRPN、F/T、G1 bridge、hold/init、shadow 和 guarded-pilot 流程，但订阅独立的
`door` 与 `door_panel` pose，并运行 573 帧 HDMI reference。默认
`--param hdmi` 使用原生 student，`--param sonic` 启动 Sonic nominal 实验。
本次实现时本机没有连接 G1、F/T 或 VRPN，因此没有声称完成真机测试。

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
