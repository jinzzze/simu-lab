# DP-D：Diffusion Policy 独立追加组

用户于 2026-10-06 请求增加 Diffusion Policy 组。冻结方案见 [diffusion_policy.json](../configs/diffusion_policy.json)。三种子模型及 192 次物理评测已完成，严格成功 84/192（43.75%）；逐种子结果与局限见[完整报告](../artifacts/reports/diffusion_policy_v1/REPORT_ZH.md)和[机器可读记录](../artifacts/reports/diffusion_policy_v1/results.json)为准。

## 与原实验的关系

原 A/B/C/D/R 研究不变。DP-D 固定使用已有 `D_seed7` 视觉定位检查点，从初始完整 RGB 得到物体 XY。个人视频通过这个视觉模型参与；扩散策略本身的动作监督来自新采集的仿真专家示范。没有把 MediaPipe 手部关键点当成机器人动作。

这个版本是**初始 RGB 状态条件的动作扩散策略**，不是端到端多帧图像 Diffusion Policy，也不是官方论文全部配置的复刻。它独立学习末端与夹爪命令；运行时不调用原固定抓取状态机。

## 数据与模型

- 仿真训练场景 50000–50127：128 条完整轨迹；验证 51000–51015：16 条。全部专家执行成功，未按结果筛除。
- 每条 320 步，25 Hz；共 46,080 个带指令的控制步。专家较早完成后，继续执行明确的末端保持命令至固定时长。
- 每个动作在物理执行前记录：限速后的绝对末端 XYZ 目标，以及夹爪目标宽度。实际物体位移不作为动作标签。
- 每个观测含初始视觉 XY、机器人末端 XYZ、实际夹爪宽度、上一有效命令和运行时钟，共 11 维。策略使用最近两个观测；初始时重复第一帧。
- 预测未来 16 步，执行前 8 步后重新读取机器人自身状态。没有在执行途中重新定位物体；目标位置固定，并隐含在专家数据中。
- 紧凑 1D U-Net，FiLM 条件残差块，宽度 64/128，共 374,084 个参数。100 步余弦噪声、噪声预测 MSE；推理为 20 步 DDIM、eta=0。
- 三个固定训练种子 7/17/27，各 5,000 步、batch 128、AdamW 1e-4；EMA 0.995，全部评测最终 EMA，没有挑最好检查点。
- 观测与动作归一化仅使用训练集；序列不会跨 episode，观测历史不含未来信息，越界未来动作不参与损失。

扩散采样保留随机初始噪声，每个测试场景固定采样种子，并在三个训练种子间配对。运行时只裁剪工作范围、夹爪范围并限制末端单步移动；这些通用执行约束不决定抓取阶段。上一命令记录的是限速后的有效末端目标，与训练语义一致。

## 评测边界

每个策略在既有 64 个测试初始场景 30000–30063 上运行 320 步（12.8 秒），三种子共 192 次。原控制器在固定阶段完成后结束，因此执行时长也未严格配对。使用相同物理参数与严格抓取/释放/稳定判据；物体真值只用于独立终态评分。每个种子固定保存第一场景 30000 视频，不根据成败挑选。

参照是既有 `D_seed7 + 固定控制器` 的 64 次执行，因为它使用完全相同的视觉模型；不把其结果重复计作 192 次新执行。DP 额外使用 128 条机器人专家轨迹且训练预算不同，因此控制器差异不能解释为扩散算法或辅助监督的单独因果效果。原测试场景再次用于追加开发评估，不是新独立测试集。

本版本依赖初始视觉估计和运行时钟，未验证执行中扰动物体、改变速度/目标、外观变化或真实机器人。离线动作误差只是诊断；最终任务成功需要实际物理评测。

## 文件与运行

代码：

- [轨迹采集](../scripts/collect_diffusion_demonstrations.py)：调用旧控制器生成专家示范，单独保存新的数据。
- [动作扩散](../src/policy/diffusion.py)：网络、因果序列、扩散训练与 DDIM 推理。
- [训练](../scripts/train_diffusion_policy.py)：全部预定种子的固定最终 EMA。
- [物理评测](../scripts/evaluate_diffusion_policy.py)：滚动动作执行与来源检查。
- [分析](../scripts/analyze_diffusion_policy.py)：完整样本、图像一致性、动作时序和哈希审计。

```powershell
Set-Location -LiteralPath E:\HumanoidRobotLearning
. .\scripts\Enter-Project.ps1
.\.env\python.exe -m pytest tests/test_diffusion_policy.py -q
.\.env\python.exe -m pytest tests/test_policy_environment.py -q
.\.env\python.exe scripts/analyze_diffusion_policy.py
```

两个测试文件须分进程运行；Torch 与 Bullet 原生库在本 Windows 环境中需要隔离。部署由持久 Torch 子进程接收初始 RGB、机器人历史和采样随机种子，输出动作序列。

首次执行的流水线入口为 `collect_diffusion_demonstrations.py` → `train_diffusion_policy.py` → `evaluate_diffusion_policy.py`。采集与训练检测到已有目录会拒绝覆盖；评测核对模型、配置和源码一致后可以续跑尚未完成的场景。已有模型与数据应保留，不为重跑删除实验工件。

方法依据：[Diffusion Policy 项目](https://diffusion-policy.cs.columbia.edu/)与[官方代码](https://github.com/real-stanford/diffusion_policy)。本实现使用现有 PyTorch，不引入额外依赖。
