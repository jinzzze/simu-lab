# 项目环境

项目根目录：`E:\HumanoidRobotLearning`。源码、数据、Python 环境、模型、缓存、临时文件和实验结果均保存在此目录。既有 Anaconda 只作为环境创建工具；本项目 Python 位于 `.env`。

## 当前实测状态

2026-10-06 11:12:58 UTC 的[环境检查](../artifacts/diagnostics/environment_current/environment.json)中，仿真渲染与机械臂移动、CUDA/ResNet18 前向、ArUco/MediaPipe 模型加载三个检查均通过。检查中的 PandaPush-v3 只用于验证渲染和指令运动；当前研究使用独立实现的抓取环境，实测结果见 [README](../README.md)。

| 项目 | 当前记录 |
|---|---|
| 系统 | Windows 10.0.26200，16 个逻辑 CPU，31.26 GiB RAM |
| Python | 3.11.16，conda-forge |
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop GPU，16.0 GiB |
| 视觉训练 | torch 2.10.0+cu126；torchvision 0.25.0+cu126；CUDA runtime 12.6 |
| 仿真 | conda-forge PyBullet 3.25；panda-gym 3.0.7；Gymnasium 0.29.1 |
| 数据处理 | NumPy 1.26.4；opencv-contrib-python 4.11.0.86；MediaPipe 1.0.1 |

PyBullet 的 conda 包版本为 3.25，Python 包元数据报告 3.2.5；这是两个记录层级，不代表同时安装了两份 PyBullet。实际来源与构建记录见 [conda 包清单](../artifacts/diagnostics/conda-packages-20261006.json)。

原环境 37 项测试通过；新安装环境另增加 2 项路径迁移测试，共 39 项通过。模型加载成功、空白图手检测返回零、测试通过都不等于真实样片上的标签准确率已知。伪标签证据范围见 [数据与标签报告](PILOT_LABEL_REPORT.md)。

## 进入现有环境

```powershell
Set-Location -LiteralPath E:\HumanoidRobotLearning
. .\scripts\Enter-Project.ps1
.\.env\python.exe --version
```

脚本只改变当前 PowerShell 进程的 PATH、缓存和临时目录，不改全局设置。运行时使用项目 `.env\python.exe`，不要直接调用另一套 Python。

## Windows 原生库隔离

当前机器已观察到完整 Torch 与 Panda/Bullet 栈在同一进程中加载时发生 `libomp.dll` / `libiomp5md.dll` 冲突。仿真在父进程运行，视觉推理由 `src/predictor_process.py` 的持久子进程完成；只传递 RGB 图像字节和预测 XY，仿真物体真值不进入预测进程。

环境检查和测试同样分进程运行，不使用 `KMP_DUPLICATE_LIB_OK`。检查当前安装时执行：

```powershell
.\.env\python.exe scripts\check_environment.py --output artifacts\diagnostics\environment_current
.\.env\python.exe scripts\run_tests.py
```

不要用单个 `pytest tests` 进程替代项目测试入口。该约束属于当前机器的运行配置，不保证其他机器会出现相同冲突。

## 新环境的安装步骤

已有 `.env` 可直接使用；以下命令仅用于需要创建新环境时。2026-10-07 已在同一台电脑建立全新环境，依赖检查、39 项测试、CUDA/视觉/仿真和单场景抓取检查通过；见 [记录](../artifacts/diagnostics/clean_install_v1/receipt.json)。另一台电脑与完整重训仍未验证。

```powershell
Set-Location -LiteralPath E:\HumanoidRobotLearning
. .\scripts\Enter-Project.ps1
conda env create --prefix .env --file environment.yml
.\.env\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.env\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
.\.env\python.exe scripts\check_environment.py --output artifacts\diagnostics\environment_current
```

Conda 需先可用；按本机安装位置调用实际可执行文件。PyBullet 由 conda-forge 管理，不通过 pip 覆盖或重新源码编译。MediaPipe 使用 Tasks API，不依赖旧的 `mp.solutions`。

[实际 pip freeze](../artifacts/diagnostics/pip-freeze-20261006.txt)仅用于记录，其中 conda 构建路径不是可用的 pip 安装地址，不能把它当作完整安装文件。外部视觉模型来源与校验值见 `configs/model_sources.json`。

## 复现范围

完整主实验入口为 `scripts/Run-PilotExperiment.ps1`，要求本地已存在两份共享 NPZ 输入；详细用法见 [复现说明](REPRODUCIBILITY.md)。原片、NPZ、权重、环境和大型视频当前保留在本地；这些输入缺失时，代码本身无法重建这一批结果。配置、来源清单与小型报告可进入版本管理，大型产物按 `.gitignore` 排除。

最初的 2026-10-03 环境检查保留在 [P1 历史记录](P1_ENVIRONMENT_REPORT.md)，不代表当前项目仍处于未采集、未训练阶段。


## Release preparation update (2026-10-07)

Independent real-scene validation is deferred by the applicant. Local source/report and input/checkpoint release candidates, an English application draft, a 112-second captioned demonstration and a 48-frame blinded development-label review are prepared. Human box/keypoint accuracy is not yet measured. Same-host clean installation passes the recorded checks; publication, final license/data permissions, applicant identity and the actual application remain pending. See [release checklist](RELEASE_CHECKLIST_ZH.md) and [archive reproduction](RELEASE_REPRODUCTION.md).
