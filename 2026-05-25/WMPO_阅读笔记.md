# WMPO: World Model-based Policy Optimization for Vision-Language-Action Models

> **论文信息**
> - 作者：Fangqi Zhu<sup>1,2</sup>, Zhengyang Yan<sup>1</sup>, Zicong Hong<sup>1</sup>, Quanxin Shou<sup>1</sup>, Xiao Ma<sup>2*</sup>, Song Guo<sup>1*</sup>
> - 通讯作者：Xiao Ma (ByteDance Seed), Song Guo (HKUST)
> - 投稿方向：ICLR 2026
> - arXiv ID：2511.09515
> - 代码：[github.com/WM-PO](https://WM-PO.github.io)（基于 verl + OpenSora，完整开源）
> - 项目页：https://wm-po.github.io/

---

## 一、核心问题

VLA（Vision-Language-Action）模型在机器人操控领域展现了巨大潜力，但存在两个根本性瓶颈：

1. **模仿学习（IL）的脆弱性**：现有 VLA 主要依赖专家演示做行为克隆，遭遇 OOD（分布外）状态时会产生级联错误，无法自纠正和从失败中学习。
2. **真实环境 RL 的采样低效**：直接在真实机器人上做 RL 需要数百万次交互，成本高、不安全、不可扩展。而构建精确仿真器又面临巨大的工程开销。

**核心矛盾**：IL 无法从失败中学习，RL 无法高效采样 —— 如何在不接触真实环境的情况下，让 VLA 策略获得 RL 的自我改进能力？

> 论文的核心洞察：利用**像素空间**（而非潜空间）的视频生成世界模型来模拟环境动力学，使 VLA 策略可以在"想象"中进行 on-policy RL，从而与预训练 VLA 的视觉表征保持一致。

![Figure 1：三种 VLA 训练范式对比 —— (a) 模仿学习只能从人类演示学习，无法处理失败和自纠正；(b) 真实环境 RL 采样成本高、难以规模化；(c) WMPO 通过预训练世界模型，在想象中实现样本高效的 on-policy RL。](assets/wmpo/introduction.jpg)

---

## 二、核心思路 / 方法

### 2.1 总体框架

WMPO 将 VLA RL 完全建立在**动作条件视频生成世界模型**之上。训练流程包含三个核心组件：

1. **想象轨迹生成（Imagined Trajectory Generation）**：策略 $\pi_\theta$ 与世界模型 $p_\phi$ 交替交互，自回归地生成完整轨迹
2. **轨迹采样（Trajectory Sampling）**：对同一初始状态采样多条轨迹，用奖励模型 $R_\psi$ 评估成功/失败
3. **策略更新（Policy Update）**：使用 GRPO（Group Relative Policy Optimization）在"想象轨迹"上优化策略参数

![Figure 2：WMPO 总体框架。从初始状态 s₀ 开始，(1) VLA 策略预测动作块 → 世界模型生成未来帧 → 交替交互形成想象轨迹；(2) 多次采样并评估；(3) 通过 GRPO 更新策略参数。](assets/wmpo/method.jpg)

### 2.2 问题形式化

将 VLA 操控任务形式化为 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R)$：

- **状态空间** $\mathcal{S} = \mathcal{I} \times \mathcal{G}$：图像观测序列 $I_{0:K}$ + 语言指令
- **动作空间** $\mathcal{A}$：动作块（action chunk），每个动作有 $D$ 个自由度，每维离散化为 256 bins
- **转移函数** $P$：由参数化世界模型 $s_{t+1} \sim p_\phi(s_{t+1} \mid s_t, a_t)$ 实现
- **奖励函数** $R_\psi$：训练好的 VideoMAE 分类器，输出轨迹是否成功的二值判断

优化目标：

$$\max_\theta \mathbb{E}_{\tau \sim \pi_\theta, p_\phi}\left[R_\psi(\tau)\right]$$

### 2.3 生成式世界模型

**架构选择**：基于 OpenSora（STDiT-v3 视频扩散模型），做出三项关键修改：

| 修改 | 动机 | 效果 |
|------|------|------|
| **3D VAE → 2D VAE（SDXL）** | 3D VAE 的时序压缩会丢失细粒度运动细节 | 更好地保留机器人-物体交互的精细运动 |
| **噪声条件帧（Noisy Frame Conditioning）** | 自回归生成中误差累积导致长期预测崩溃 | 训练时对条件帧加 50/1000 步扩散噪声，提升鲁棒性 |
| **帧级动作控制（Frame-Level Action Control）** | 需要精确的动作-帧对齐 | 扩展 AdaLN 块，以帧粒度注入动作信号 |

**帧级动作控制**的具体实现：对每帧 $i$ 的动作 $a_i$，MLP 生成调制系数 $\gamma_1^i$、$\beta_1^i$（LayerNorm 的 scale/shift）、$\alpha_1^i$（残差连接的 scale）：

$$\mathbf{x}^i = \mathbf{x}^i + (1 + \alpha_1^i) \cdot \text{Block}\bigl(\gamma_1^i \cdot \text{LayerNorm}(\mathbf{x}^i) + \beta_1^i\bigr)$$

**Policy Behavior Alignment（策略行为对齐）**：这是 WMPO 的关键创新之一。世界模型先在 OXE（Open X-Embodiment，百万级机器人轨迹）上预训练，再在下游任务的策略自采数据上微调。这一步解决了专家演示与策略 rollout 之间的状态分布不匹配问题 —— 世界模型需要见过策略自己的失败模式才能忠实模拟。

### 2.4 奖励模型

- **架构**：VideoMAE-base 编码器 + 线性分类头
- **训练**：二分类交叉熵，以轨迹末端 clip 为正样本，轨迹中间 clip 和失败轨迹 clip 为负样本；batch 内正负样本平衡
- **推理**：对整个想象轨迹以 stride=1 滑动窗口扫描，任一时序窗口的预测概率超过阈值 $\tau_{\text{thr}}$ 即判定成功
- **效果**：所有任务上 F1 > 0.95

### 2.5 On-Policy GRPO 优化

采用 GRPO（Group Relative Policy Optimization），关键设计选择：

- **组内相对优势**：从同一初始状态采样 $G=8$ 条轨迹，用组内标准化计算优势：$\hat{A}_i = \frac{R_i - \text{mean}(\{R_i\})}{\text{std}(\{R_i\})}$
- **去除 KL 散度正则**（遵循 DAPO）：不使用参考模型，减少显存消耗，鼓励策略探索
- **动态采样（Dynamic Sampling）**：若一组轨迹全部成功或全部失败，丢弃该组重新采样 —— 避免梯度消失
- **不对称裁剪**：$\epsilon_{\text{low}}=0.20$，$\epsilon_{\text{high}}=0.28$

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{T}\sum_{t=0}^{T} \min\!\Big(r_{i,t}(\theta)\hat{A}_i, \operatorname{clip}(r_{i,t}(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\hat{A}_i\Big)\right]$$

---

## 三、实验与结果

### 3.1 实验设置

- **模拟环境**：Mimicgen，4 个精细操控任务：Coffee_D0、StackThree_D0、ThreePieceAssembly_D0、Square_D0
- **基础策略**：OpenVLA-OFT 在每任务 300 条专家演示上微调
- **Rollout 预算**：$P=128$ 和 $P=1280$，代表可用于优化的真实轨迹数量
- **评估**：每任务 128 个不同初始状态，报告平均成功率
- **GPU**：SFT 8×H100，世界模型和策略优化 32×H100

### 3.2 主要对比结果

| Rollout 预算 | 方法 | Coffee | StackThree | ThreePieceAssembly | Square | 平均 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| *—* | *Base Policy (SFT)* | 43.8 | 46.9 | 19.5 | 24.2 | 33.6 |
| 128 | GRPO | 38.3 | 52.3 | 17.2 | 25.0 | 33.2 |
| 128 | DPO | 43.8 | 53.9 | 23.4 | 28.1 | 37.3 |
| 128 | **WMPO** | **61.7** | **56.3** | **37.5** | **32.8** | **47.1** |
| 1280 | GRPO | 47.7 | 54.7 | 20.3 | 25.8 | 37.1 |
| 1280 | DPO | 52.3 | 57.0 | 26.7 | 33.6 | 42.4 |
| 1280 | **WMPO** | **75.0** | **64.1** | **46.1** | **45.3** | **57.6** |

**关键发现**：

1. **小预算高效**：$P=128$ 时 WMPO 比最强基线 DPO 高 +9.8 个百分点，比 base policy 高 +13.5 个百分点
2. **扩展性强**：预算从 128 增至 1280 时，WMPO 提升 +10.5 个百分点（47.1→57.6），DPO 仅提升 +5.1（37.3→42.4）
3. **GRPO 在真实环境中受限**：$P=128$ 时 GRPO 甚至低于 base policy（33.2 vs 33.6），因为在线 GRPO 需要大量实时交互，小预算下采样不足

### 3.3 涌现行为：自纠正

![Figure 3：Square 任务（将方块插入立柱）的行为分析。基础策略（上排）碰到立柱后持续推挤直到超时失败；WMPO（下排）学会了抬起方块、重新对齐、正确插入——这是一从未在演示数据中出现过的自纠正行为。](assets/wmpo/behavior.jpg)

**自纠正行为分析**：

- **基策略行为**：当方块偏离正确轨迹与立柱碰撞时，基策略只知道继续向前推（模仿所见过的演示），最终超时失败
- **WMPO 行为**：世界模型生成的"想象轨迹"中包含了大量碰撞、偏离等失败情形，策略通过学习哪些动作序列能获得成功奖励，涌现出"抬起→重新对齐→再插"的自纠正策略
- **为什么重要**：这种能力无法通过单纯的模仿学习获得，因为它需要见过失败并从中学习恢复 —— 这正是 RL 的核心价值

### 3.4 轨迹效率提升

![Figure 5：不同策略成功轨迹的相对平均长度（Base Policy = 100%）。WMPO 训练的轨迹显著更短，因为策略学会避免"卡住"的次优状态，不仅提高成功率也提升了执行效率。](assets/wmpo/length.jpg)

WMPO 成功轨迹的平均长度明显短于基策略。原因是 WMPO 抑制了"卡住"（stuck）行为 —— 该行为在 IL 中常见，不仅降低效率，还经常导致超时失败。更短的轨迹意味着策略执行更流畅、更高效。

### 3.5 泛化能力

WMPO 在三种分布偏移（Distribution Shift）下测试泛化：

<table><tr>
<td width="33%"><img src="assets/wmpo/generalization_a.jpg" width="100%"><br><em>(a) 位置偏移（Position Disruption）：Square 任务，立柱位置从固定变为矩形内随机</em></td>
<td width="33%"><img src="assets/wmpo/generalization_b.jpg" width="100%"><br><em>(b) 背景偏移（Background Disruption）：StackThree 任务，桌面换为灰色背景</em></td>
<td width="33%"><img src="assets/wmpo/generalization_c.jpg" width="100%"><br><em>(c) 纹理偏移（Texture Disruption）：ThreePieceAssembly 任务，红色底座换为深色木纹</em></td>
</tr></table>

| 方法 | 位置偏移 | 背景偏移 | 纹理偏移 | 平均 |
|------|:---:|:---:|:---:|:---:|
| Base Policy | 14.1 | 46.1 | 10.9 | 23.7 |
| GRPO | 15.6 | 47.7 | 10.9 | 24.7 |
| DPO | 16.4 | 34.4 | 7.8 | 19.5 |
| **WMPO** | **22.3** | **50.0** | **16.4** | **29.6** |

**关键发现**：DPO 虽然在分布内有所提升，但在背景和纹理偏移下**退化**（甚至低于 base policy），说明它可能学到了对特定视觉线索的过拟合；WMPO 在三种偏移下全部最优，证明了在世界模型中训练有助于学到更泛化的操控技能。

### 3.6 终身学习（Lifelong Learning）

![Figure 6：StackThree 任务上的终身学习。WMPO 通过迭代收集真实轨迹→世界模型优化→更新策略→重新收集，实现了稳定的持续改进。DPO 无法迭代提升（训练不稳定），而 base policy 即使用更多专家演示也提升有限。](assets/wmpo/lifelong.jpg)

- 每轮收集 $P=128$ 真实轨迹 → WMPO 优化策略 → 用新策略收集下一轮数据
- WMPO 三轮后稳定提升，DPO 由于离线训练的固有不稳定性无法迭代改进
- Base Policy 即使使用 556 条专家演示（vs 基础 300 条），提升也远不及 WMPO —— 说明自动探索比增加演示更有效

### 3.7 真机实验

![Figure 7：真机实验（"将方块插入立柱"，间隙仅 5mm）。上排：基策略在真实环境中的执行；下排：世界模型对同一初始状态的"想象"预测。尽管从未见过这条轨迹，世界模型准确预测了未来演化。Base / DPO / WMPO 的 30 次试验成功率分别为 53% / 60% / 70%。](assets/wmpo/real_success.jpg)

- **平台**：Cobot Mobile ALOHA
- **任务**：高精度插入（方块与立柱间隙仅 5mm）
- **结果**：Base 53% → DPO 60% → WMPO **70%**
- **世界模型保真度**：即使在从未见过的场景中，世界模型也能准确预测物体交互和运动

---

## 四、关键洞察与技术亮点

1. **像素空间 > 潜空间**：WMPO 坚持在像素空间生成视频，而非像 Dreamer 系列那样在潜空间学动力学。理由是 VLA 的视觉编码器用海量互联网图片预训练，包含丰富的语义理解 —— 在潜空间中重新训练会导致表征失配

2. **Policy Behavior Alignment 是必需组件**：仅用专家演示训练世界模型无法模拟策略可能遇到的失败状态，必须用策略自身的 rollout 数据微调世界模型

3. **噪声条件帧**：自回归视频预测中，先前生成的帧带有伪影和误差，使用干净的帧作为条件会导致训练-推理分布不一致。给条件帧加噪声（50/1000 步扩散噪声）让模型学会容忍不完美的条件输入

4. **GRPO + 世界模型 = 天然匹配**：GRPO 要求从同一初始状态采样多条轨迹来计算组内相对优势，这在真实世界中几乎不可能（无法精确复现同一初始状态），但在世界模型中可以轻松重复采样

5. **自纠正 = RL 价值的体现**：WMPO 涌现的自纠正行为是论文最有力的定性证据 —— 策略学会了演示数据中完全没有的动作模式（抬起→对齐→再插），证明了 RL 相对于 IL 的本质优势

6. **终身学习闭环**：WMPO 支持策略→收集→世界模型更新→策略优化的迭代循环，这是真实环境中难以实现的

---

## 五、代码实现解读

WMPO 代码基于三个开源项目构建：**verl**（RL 训练框架）、**OpenSora**（世界模型）、**OpenVLA-OFT**（VLA 策略）。

### 5.1 系统架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           WMPO Training System                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────── Ray Cluster ───────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐     │   │
│  │  │ RobWMActorRollout│   │  CriticWorker   │   │ RobWMActorRollout│     │   │
│  │  │   Worker (GPU)   │   │    (GPU)         │   │   (RefPolicy)    │     │   │
│  │  │                  │   │                  │   │                  │     │   │
│  │  │ ┌─────────────┐  │   │ ┌─────────────┐  │   │ ┌─────────────┐  │     │   │
│  │  │ │ VLA Policy  │  │   │ │Value Network│  │   │ │ Old Policy  │  │     │   │
│  │  │ │ (OpenVLA)   │  │   │ │  (optional)  │  │   │ │  (frozen)   │  │     │   │
│  │  │ └─────────────┘  │   │ └─────────────┘  │   │ └─────────────┘  │     │   │
│  │  │ ┌─────────────┐  │   │                  │   │                  │     │   │
│  │  │ │ World Model │  │   │                  │   │                  │     │   │
│  │  │ │(OpenSora)   │  │   │                  │   │                  │     │   │
│  │  │ └─────────────┘  │   │                  │   │                  │     │   │
│  │  │ ┌─────────────┐  │   │                  │   │                  │     │   │
│  │  │ │ Reward Model│  │   │                  │   │                  │     │   │
│  │  │ │ (VideoMAE)  │  │   │                  │   │                  │     │   │
│  │  │ └─────────────┘  │   │                  │   │                  │     │   │
│  │  └─────────────────┘   └─────────────────┘   └─────────────────┘     │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  关键文件映射:                                                                 │
│  ├── verl/trainer/main_ppo.py            # 训练入口，配置 Ray + Worker         │
│  ├── verl/workers/rollout/robwm_rollout.py  # 世界模型 rollout（核心）         │
│  ├── verl/workers/rollout/rob_rollout.py    # 真实环境 rollout（baseline）      │
│  ├── verl/workers/fsdp_workers.py        # FSDP 分布式 Worker 实现             │
│  ├── verl/trainer/ppo/core_algos.py      # GRPO 核心算法                       │
│  ├── verl/trainer/ppo/ray_trainer.py     # Ray 分布式训练调度                   │
│  └── reward_model/videomae.py            # 奖励模型训练                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 世界模型 Rollout 流程（robwm_rollout.py）

这是 WMPO 的核心代码，`RobWMHFRollout` 类实现了"在想象中做 RL"的完整流程：

```
  初始状态 s₀ ────┐
                 │
   ┌─────────────▼─────────────┐
   │ 1) encode(s₀) → VAE latent│
   │    image_history = repeat │
   │      (latent, queue_len)  │
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │ 2) VLA π_θ(current_frame) │   _generate_one_step()
   │    → action chunk (8 dim)  │   论文: 公式 π_θ(a_t|s_t)
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │ 3) World Model p_φ:       │   scheduler.sample(
   │    z = concat(history,     │     model=world_model,
   │         noise)             │     y=actions,
   │    latent = denoise(z|     │     mask=[0..0,1..1])
   │             action)        │   论文: 公式 I~p_φ(I|c,a)
   │    frame = vae.decode()    │
   └─────────────┬─────────────┘
                 │
                 ▼  重复直到 max_steps
                 │
   ┌─────────────▼─────────────┐
   │ 4) Reward Model R_ψ:      │   predict_success()
   │    slide_window(video)     │   滑动窗口 L=w=8, stride=1
   │    → complete / fail       │   论文: 二值分类 R_ψ(τ)∈{0,1}
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │ 5) GRPO Update:            │   core_algos.py
   │    adv = (R-mean)/std       │   论文: 公式 Eq.(4)
   │    loss = clip_ratio * adv  │
   └───────────────────────────┘
```

**代码关键函数映射**：

| 论文公式/概念 | 代码位置 |
|---|---|
| 想象轨迹生成 (Eq.1) | `robwm_rollout.py:run_wm_inference()` |
| 世界模型条件扩散 | `robwm_rollout.py:339-395`（scheduler.sample + vae.decode） |
| 帧级动作控制 (Eq.2) | OpenSora 的 AdaLN 扩展（在 world model 的 transformer block 中） |
| 奖励模型推理 | `robwm_rollout.py:predict_success()` |
| GRPO 目标 (Eq.3+4) | `core_algos.py` + `ray_trainer.py` |
| 动态采样（DAPO 策略） | `ray_trainer.py`（过滤全成功/全失败组） |
| Policy Behavior Alignment | 世界模型在 `train_wmpo_*.sh` 的训练脚本中实现 |

### 5.3 奖励模型训练（videomae.py）

- **数据构造**：`SuccessWindowDataset` 类从 webdataset tar 中提取滑动窗口 clips
  - 正样本：轨迹末端 $[T-W, T]$ 的 clip（label=1，仅成功轨迹）
  - 负样本：轨迹中间窗口（$[T-S, W-1]$ 范围）的 clips + 失败轨迹的任意 clips
- **训练**：DDP，每 1000 步验证一次，grid search 确定最优阈值 $\tau_{\text{thr}}$

### 5.4 双模式 Rollout

WMPO 代码同时支持两种 rollout 模式：

| 模式 | Worker 类 | 用途 |
|------|-----------|------|
| **世界模型 Rollout** | `RobWMActorRolloutRefWorker` (继承 `RobWMHFRollout`) | WMPO 主线：在"想象"中训练 |
| **真实环境 Rollout** | `RobActorRolloutRefWorker` (继承 `RobHFRollout`) | 基线（GRPO）：在 Mimicgen 仿真中直接交互 |

切换方式：配置 `actor_rollout_ref.wm.enable = true/false`（见 `main_ppo.py:153`）。

---

## 六、局限性

1. **动作表征**：当前仅支持离散化动作（每维 256 bins），未扩展到 flow-matching 等连续/更表达性的策略类（如 π₀）
2. **计算开销**：世界模型预训练需 1200 万步（32×H100），总体资源需求相当可观。不过论文指出，相比真实机器人 rollout 的成本（不可扩展、有安全风险），算力是可灵活扩展的资源
3. **基线覆盖**：仅对比 GRPO 和 DPO，缺少与 DreamerV3 等潜空间世界模型方法的对比。但这部分是由于现有潜空间方法不直接兼容 VLA 设置

---

## 七、关键概念速查

| 概念 | 说明 |
|------|------|
| **WMPO** | 在视频生成世界模型中进行 on-policy VLA RL，无需真实环境交互 |
| **Policy Behavior Alignment** | 用策略自己的 rollout 数据微调世界模型，解决专家演示与策略行为的状态分布不匹配 |
| **Noisy Frame Conditioning** | 训练时对条件帧加扩散噪声，提升自回归预测的鲁棒性 |
| **Frame-Level Action Control** | 扩展 AdaLN，以帧粒度注入动作信号，确保长序列中的动作-帧对齐 |
| **VideoMAE Reward Model** | 以滑动窗口扫描想象视频，输出轨迹成功/失败的二值判断（F1>0.95） |
| **GRPO (Group Relative Policy Optimization)** | 从同一初始状态采样多条轨迹，用组内相对优势做策略优化 |
| **Dynamic Sampling (DAPO)** | 丢弃全成功/全失败的轨迹组（梯度为零），重新采样 |
| **SDXL 2D VAE** | 替代 OpenSora 的 3D VAE，保留更好的细粒度运动细节 |
| **Mimicgen** | 仿真基准，包含 4 个精细操控任务 |
| **OXE** | Open X-Embodiment 数据集，用于世界模型预训练 |
