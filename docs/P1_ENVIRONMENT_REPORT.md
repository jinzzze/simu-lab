> **历史记录：以下内容保留 2026-10-03 的检查时点。** 截至 2026-10-06，12 条样片、真实视频预训练和物理抓取评测已经完成；本文末尾“尚未采集/训练”的说明不再代表当前状态。最新环境检查位于 [environment_current/environment.json](../artifacts/diagnostics/environment_current/environment.json)，使用独立进程处理 Windows 原生 OpenMP 冲突。当前环境用法见 [ENVIRONMENT.md](ENVIRONMENT.md)，任务与实测结果见 [CURRENT_TASK.md](CURRENT_TASK.md)。

# P1 环境与工具验收记录

检查时间：2026-10-03T00:41:50Z。项目：`E:\HumanoidRobotLearning`。

## 实测结果

| 检查 | 结果 |
| --- | --- |
| Python | 3.11.16，独立环境位于E盘 `.env` |
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop GPU，16.0 GiB显存 |
| RAM / CPU | 31.26 GiB / 16 逻辑处理器 |
| PyTorch | 2.10.0+cu126；CUDA矩阵运算校验通过 |
| ResNet-18 | 官方IMAGENET1K_V1权重在GPU完成前向推理 |
| PandaPush-v3 | 512×512 RGB渲染通过，指令驱动末端移动约5.23cm |
| MediaPipe | 1.0.1；官方手模型加载与空白图像处理通过 |
| OpenCV标记 | 初始四标记版已验收；2026-10-03按用户要求改为双标记版，数字布局与实际SVG预览均检测到ID 0、2 |
| 依赖一致性 | pip check通过 |

## 证据

- `artifacts/diagnostics/environment.json`：完整实测参数。
- `artifacts/diagnostics/panda_initial.png` 与 `panda_after_motion.png`：实际仿真渲染。
- `artifacts/diagnostics/pip-freeze.txt` 与 `conda-explicit-win64.txt`：实际安装清单；pip freeze仅作记录，不能用于覆盖Conda的PyBullet。
- `assets/calibration/board_geometry.json`：打印板毫米坐标和数字识别检查。
- `configs/model_sources.json`：外部模型来源与校验值。

## 还没有验证的内容

申请人的实拍视频质量、手部关键点准确率、真实物体跟踪误差、打印后实际比例、推杆动作可复现性。当前没有训练研究模型，没有测量任何四组成功率。

P1环境部分已完成，采集部分等待12次试采。拍摄说明见 `PILOT_CAPTURE.md`，记录表见 `data/manifests/pilot_plan.csv`，原片存入 `data/raw/pilot/`。
