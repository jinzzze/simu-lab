# 抓取宏动作世界模型：独立扩展

2026-10-06：数据采集、固定预算训练、真实状态输入评估、一个预选 RGB 集成演示及复用命令的 RGB 批量诊断已完成。本扩展独立于 A/B/C/D/R 视觉辅助监督实验；该主实验的模型、固定控制器、评分及 1,920 次执行结果均不改变。

## 模型学到了什么

输入为执行前物体 XY 与预先下发的绝对抓取 XY。一个 4→64→64→4 的 ReLU MLP 预测一次完整固定抓取技能结束后的二维位移、合格双指抬升概率和严格任务成功概率。输入含位置与指令相对位置差；放置目标、方块、技能及物理参数固定。

这是学习到的宏动作状态转移与结果预测器。它不生成视频、不输出语言条件动作、不替代物理引擎，也没有用来选择动作或在线重新规划。手机视频只训练上游视觉编码器；本模型的全部转移标签来自 PyBullet 仿真。

## 数据与固定训练

[配置](../configs/grasp_world_model.json)在采集和观察指标前固定。每场景执行五个预先确定动作：一个零偏移、四个在 XY 各 ±30 mm 内均匀采样的偏移。动作指令在执行前记录，不能从最终物体运动倒推。

| 划分 | 独立初始场景 | 宏动作转换 | 场景种子 |
|---|---:|---:|---|
| 训练 | 64 | 320 | 40000–40063 |
| 验证 | 16 | 80 | 41000–41015 |
| 测试 | 32 | 160 | 42000–42031 |

共 560 次物理执行，按完整初始场景划分。同场景五个动作相关，160 个测试转换不等于 160 个独立环境。训练种子 41/43/47 各固定 2,000 步、batch 64、AdamW 1e-3；保留全部最终检查点，预先约定等权平均位移和概率。

## 真实状态输入的测试结果

以下训练和评估均输入仿真真实初始 XY，衡量给定状态时的宏动作预测能力。末态位置误差等同于在真实初始状态上叠加预测位移后的 XY 误差。

| 预测器 | 平均 XY 误差 | P90 XY 误差 | 成功标签分类正确 | 成功概率 Brier |
|---|---:|---:|---:|---:|
| 零位移基线 | 90.8853 mm | 190.1115 mm | 89/160 | 0.24746 |
| 训练平均位移基线 | 92.0621 mm | 122.1010 mm | 89/160 | 0.24746 |
| 三模型等权平均 | **28.4107 mm** | **80.2912 mm** | **153/160** | **0.03646** |

分类阈值为 0.5。153/160（95.625%）是预测成功/失败标签的准确率；实际测试执行成功的是 71/160，不能把分类准确率写成机器人成功率。基线分类概率使用训练集事件比例，没有使用测试标签拟合常数。

三个单模型的平均 XY 误差分别为 29.8271、31.6303、28.3314 mm。集成抬升分类也是 153/160；成功 AUROC 为 0.9921。概率误差和排序表现是本分布的描述性指标，不足以说明真实机器人概率校准。

图表、逐种子数字和来源见[完整指标](../artifacts/reports/grasp_world_model_v1/results.json)及[实验报告](../artifacts/reports/grasp_world_model_v1/REPORT_ZH.md)。平均误差仍超过方块 20 mm 宽度，P90 达 80.29 mm；不能仅凭分类表现把模型称为精确控制器。

## 已完成的单例 RGB 集成

配置预先指定 scene 43000 与视觉模型 D_seed7。完整 RGB 经隔离视觉进程预测初始 XY，将该估计同时作为宏模型状态和抓取指令。宏模型预测先写盘，随后才执行物理动作；真实位置仅在此后用于评分。

- 初始视觉误差：1.7746 mm。
- 预测最终 XY 误差：0.7228 mm。
- 实际执行满足严格成功判据。
- 预测成功概率：0.999226；这只是一个输出，不能由一次成功验证其校准。

[输入与实际结果](../artifacts/reports/grasp_world_model_v1/integration/result.json) · [演示视频](../artifacts/reports/grasp_world_model_v1/integration/rgb_world_model_demo.mp4)

这一例说明 RGB→状态估计→结果预测→物理执行的接口可以连通。它不是总体 RGB 评测，不证明世界模型提高了执行成功率；没有新增规划或修改动作。

## 已完成的 RGB 批量输入诊断

随后冻结[追加配置](../configs/grasp_world_model_rgb_evaluation.json)，使用同一预选视觉检查点 D_seed7 重渲染 32 个测试初始场景，全部 RGB 哈希与原采集一致。输入 RGB 估计 XY 和原来的五个绝对命令，预测全部 160 个结果；没有训练、选新动作或新增物理执行。

| 初始状态输入 | 终点平均误差 | P90 | 成功标签分类正确 | 成功概率 Brier |
|---|---:|---:|---:|---:|
| 真实 XY 参照 | 28.4107 mm | 80.2912 mm | 153/160 | 0.03646 |
| RGB 估计 XY | 28.2960 mm | 77.6660 mm | 153/160 | 0.03346 |

视觉定位在 32 个场景上的平均误差为 1.2586 mm。结果见 [RGB 诊断](../artifacts/reports/grasp_world_model_v1/rgb_evaluation/REPORT_ZH.md)与[完整 JSON](../artifacts/reports/grasp_world_model_v1/rgb_evaluation/results.json)。微小差异不能解释为 RGB 比真实状态更好。

原绝对命令是在原采集阶段由真实位置加预定偏移产生；这里仅原样重放，不能称为 RGB 策略自主选择动作。这是看过真实状态结果与单例演示后定义的探索诊断，仅一个视觉检查点、复用固定外观测试分布，不证明校准、规划收益、A/B/C/D 世界模型差异或独立真实泛化。

## 运行与恢复记录

当前 Windows 环境中，Torch 与完整仿真/部分科学计算原生依赖需要分进程使用。运行前进入项目环境：

```powershell
Set-Location -LiteralPath E:\HumanoidRobotLearning
. .\scripts\Enter-Project.ps1
```

首次训练的三个最终检查点已保存；随后因原生运行库冲突，将预测导出和报告生成拆开恢复，**额外训练步数为零**。原训练源码保存在 `artifacts/runs/grasp_world_model_v1/source_snapshots/train_grasp_world_model_before_process_fix.py`；训练 manifest 的源码哈希对应该历史版本，当前脚本已改成独立报告进程。报告另外记录后处理源码哈希。

从现有检查点重新导出并生成派生报告：

```powershell
.\.env\python.exe scripts\export_grasp_world_model_predictions.py
.\.env\python.exe scripts\report_grasp_world_model.py
```

export 读取已有三个 `checkpoint.pt`，重写 `torch_predictions.npz` 与 `run_state.json`，不训练、不覆盖权重。report 读取数据与导出预测，检查 NumPy/Torch 一致性，并重写派生 `results.json`、预测文件与图表。这两个脚本使用固定目录，没有输出目录参数，运行即更新这些派生产物。实测导出最大位移差 2.98e-8 m、最大概率差 1.19e-7。

以下入口也使用固定路径，仅适用于对应输出尚不存在的首次运行：

```powershell
.\.env\python.exe scripts\generate_grasp_world_model.py
.\.env\python.exe scripts\train_grasp_world_model.py
.\.env\python.exe scripts\demo_grasp_world_model.py
.\.env\python.exe scripts\evaluate_world_model_rgb.py
```

采集遇到既有 manifest/分割日志、训练遇到既有 run 目录、演示遇到既有 `integration/result.json`、RGB 诊断遇到既有 `rgb_evaluation/results.json` 时均拒绝覆盖。当前目录已经完成，因此不要为重复运行删除原始证据；查看现有结果，或先为未来复现设计独立输出路径。演示用持久视觉子进程和 NumPy 宏模型，不使用 `KMP_DUPLICATE_LIB_OK`。

## 科学局限

转移数据全由同一仿真生成；对象、目标、朝向、相机、技能及物理参数固定。尚未验证新物体、变化目标、长时序滚动、视频预测或真实机器人。真实状态输入的测试误差不能外推为带感知误差的 RGB 表现。当前基线较简单，尚缺直接使用抓取偏移的几何规则等更强参照。三个种子重复使用同一测试集，不应按 480 个独立场景给置信区间。

已完成的 RGB 批量诊断覆盖单一视觉检查点与相同外观分布；更强感知误差、新场景和模型引导规划仍未验证。
