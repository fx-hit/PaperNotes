# World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation

> **论文信息**
> - 作者：Zhennan Jiang$^{*}$, Kai Liu$^{*}$, Yuxin Qin, Shuai Tian, Yupeng Zheng, Mingcai Zhou, Chao Yu, Haoran Li$^{\dagger}$, Dongbin Zhao
> - 单位：中国科学院自动化研究所 / 中国科学院大学 / 中关村学院 / 北京中科慧灵机器人 / 清华大学
> - 通讯作者：Haoran Li
> - 投稿方向：IEEE 期刊（IEEEtran 模板）
> - arXiv ID：2509.19080v2
> - 代码：未公开

---

## 一、核心问题

机器人操作策略通常通过**模仿学习（Imitation Learning）**初始化，但其性能受限于专家数据的**稀缺性和覆盖范围狭窄**。虽然强化学习（RL）可以通过交互来改进策略，但存在两难困境：

- **真机 RL 训练**：交互成本高、安全风险大
- **仿真器 RL 训练**：存在 sim-to-real gap

本文探索的问题是：**能否用扩散世界模型（Diffusion World Model）作为高保真仿真器，让策略完全在"想象环境"中完成 RL 优化，从而同时避免真机风险和仿真偏差？**

> 核心洞察：将扩散模型用作"可学习的仿真器"，先预训练世界模型捕获多样动态，再冻结世界模型并在其中进行端到端策略优化——全程无需真实环境交互。

---

## 二、核心思路 / 方法

### 2.1 整体框架：两阶段范式

World4RL 包含三个核心组件，分为两个阶段训练：

**组件**：
1. **Diffusion Transition Model** $D_\theta$：基于历史观测和动作预测未来观测的动态近似器
2. **Reward Classifier** $C_\psi$：二分类器，判断当前状态是否为成功状态，提供稀疏奖励信号
3. **RL-refined Policy** $\pi_\xi$：以 BC 策略为初始化的 PPO 优化策略

![图 1：World4RL 框架总览](arXiv-2509.19080v2/fig/framework.jpg)

*图 1：World4RL 框架整体架构，采用上下两栏分别展示 Stage 1（Pre-training）和 Stage 2（Policy Optimization）的完整数据流。*

*上半栏（Stage 1）从左到右有三个训练分支：① 最左侧，从 Task-Agnostic Dataset 中采样多任务数据，通过扩散去噪过程训练 Diffusion Transition Model——模型以历史帧 $x_{t-T:t}$ 和 two-hot 编码动作 $z_{t-T:t}$ 为条件，学习预测下一帧 $x_{t+1}$，使用 EDM 预条件去噪目标；② 中间，利用 Success-Annotated Dataset（含专家演示正例和 BC-policy rollout 负例/近成功例）训练 Reward Classifier（ResNet18），输出二值成功概率 $r \in \{0,1\}$；③ 最右侧，从 Expert Demonstrations 中通过极大似然估计训练 BC Policy，输出高斯分布 $\mathcal{N}(\mu, \Sigma)$。三个组件共享部分视觉编码器但各自独立优化。*

*下半栏（Stage 2）展示 Policy Optimization 的循环：当前观测 $x_t$ 输入策略网络，输出动作分布并采样 $a_t$，经 Two-Hot Encoding 转换为 $z_t$；历史 $T$ 帧观测和编码动作拼接为条件 $c$，与噪声化的目标帧 $x^\tau_{t+1}$ 一起输入冻结的 Diffusion Transition Model，通过迭代去噪生成预测帧 $\tilde{x}_{t+1}$；Reward Classifier 对 $\tilde{x}_{t+1}$ 输出成功/失败信号 $r$；$(x_t, a_t, r, \tilde{x}_{t+1})$ 存入 buffer，攒够 batch 后 PPO 同时更新 Policy 和 Value Network。全过程用虚线框标注"Frozen"，强调世界模型和奖励分类器不参与梯度更新。图底部标注了各组件使用的数据源（Experts / Rollouts / Random），以及数据流方向（实线箭头 = 前向推理，虚线箭头 = 梯度更新）。*

### 2.2 Stage 1：预训练阶段

#### (a) 策略预训练（BC）

策略 $\pi_\xi$ 输出高斯分布 $\mathcal{N}(\mu_\xi(x_t), \Sigma_\xi(x_t))$，通过最大化专家动作的对数似然进行训练：

$$\mathcal{L}_{BC}(\xi) = -\mathbb{E}_{(x_t, a_t)\sim D_{exp}}[\log \pi_\xi(a_t | x_t)]$$

BC 初始化确保 RL 阶段从一个合理起点出发，避免稀疏奖励环境中的探索困难。

#### (b) 奖励分类器

使用 ResNet18 作为视觉骨干，对观测 $x_{t+1}$ 输出二值成功概率：

$$\mathcal{L}_C(\psi) = -\frac{1}{N}\sum_{i=1}^N [r_i \log C_\psi(x_i) + (1-r_i)\log(1-C_\psi(x_i))]$$

训练数据同时包含专家演示（正例为主）和 BC 策略 rollout（引入近成功/失败/偏离状态），提高分类器区分能力。

#### (c) 扩散转移模型

基于 EDM（Elucidating Diffusion Models）的预条件去噪公式：

$$D_\theta(x^\tau; \tau, c) = c_{skip}^\tau x^\tau + c_{out}^\tau F_\theta(c_{in}^\tau x^\tau; c_{noise}^\tau, c)$$

条件 $c$ 包含：过去 $T$ 帧观测 $x^0_{t-T:t}$ 和对应的 two-hot 编码动作 $z_{t-T:t}$。

**训练数据混合**（关键设计）：
- 专家演示 $D_{exp}$：提供高质量成功轨迹
- BC 策略 rollout $D_{rollout}$：覆盖策略执行时的状态分布
- 随机 rollout $D_{rand}$：扩大状态-动作覆盖范围，防止 OOD 退化

### 2.3 Two-Hot 动作编码

针对机器人操作中连续高维动作空间的需求，采用**双热编码**（灵感来自 DreamerV3）：

对每个动作维度 $a_i$，找到最近的两个 bin $b_k, b_{k+1}$：

$$\mathbf{t}_i[k] = \frac{b_{k+1} - a_i}{b_{k+1} - b_k}, \quad \mathbf{t}_i[k+1] = \frac{a_i - b_k}{b_{k+1} - b_k}$$

其中 $\sum_j \mathbf{t}_i[j] = 1$，$K=21$ 个 bin。

**优势**：
- 无损失、可微分的连续-离散混合表示
- 相比 one-hot 更细粒度，相比 VQ-VAE/FAST 无重建误差
- 可与策略网络端到端优化

### 2.4 Stage 2：策略优化阶段

冻结 $D_\theta$ 和 $C_\psi$，仅优化 $\pi_\xi$ 和价值网络 $V_\phi$，使用 PPO 算法：

$$\mathcal{L}_P(\xi) = \mathbb{E}_t [\min(\rho_t(\xi) A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)]$$

$$\mathcal{L}_V(\phi) = \mathbb{E}_t [(V_\phi(x_t) - (r_t + \gamma V_\phi(x_{t+1})))^2]$$

**每次交互循环**：
1. 观测 $x_t$ → 策略采样 $a_t$ → two-hot 编码为 $z_t$
2. 扩散模型生成 $\tilde{x}_{t+1} = D_\theta(\cdot | x_{t-T:t}, z_{t-T:t})$
3. 奖励分类器给出 $r_t = C_\psi(\tilde{x}_{t+1}) \in \{0, 1\}$
4. 收集足够轨迹后 PPO 更新

**受控探索（关键设计）**：PPO 训练中限制策略标准差 $\sigma \le e^0$（常规 PPO 允许 $\sigma \le e^2$），使采样动作保持在世界模型训练分布的支持范围内，减少 OOD 生成。

---

## 三、训练目标

| 阶段 | 组件 | 损失函数 | 数据 |
|------|------|----------|------|
| 预训练 | BC 策略 | $\mathcal{L}_{BC}(\xi)$ — 负对数似然 | $D_{exp}$（专家演示） |
| 预训练 | 奖励分类器 | $\mathcal{L}_C(\psi)$ — 二值交叉熵 | $D_{exp} \cup D_{rollout}$ |
| 预训练 | 扩散转移模型 | $\mathcal{L}_D(\theta)$ — EDM 去噪损失 | $D_{exp} \cup D_{rollout} \cup D_{rand}$ |
| 策略优化 | PPO | $\mathcal{L}_P(\xi) + \mathcal{L}_V(\phi)$ | 世界模型想象 rollout（无需真实数据） |

---

## 四、实验与结果

### 4.1 实验设置

- **仿真平台**：Meta-World benchmark（6 个任务：coffee-pull-v2, soccer-v2, hammer-v2, door-lock-v2, lever-pull-v2, handle-pull-v2）
- **世界模型训练数据**（每任务）：50 条专家轨迹 + 150 条 BC-policy rollout + 30 条随机 rollout，每条 50 步
- **模型规模**：World4RL 330M 参数（与 NWM 320M 相当，小于 iVideoGPT 430M，远大于 DiWA 40M）
- **真机平台**：Franka Emika Panda 机械臂，6 个操作任务

### 4.2 世界模型保真度（视频预测）

**表 1：视频预测量化结果**

| 模型 | FVD↓ (Policy) | FVD↓ (Random) | FID↓ (Policy) | FID↓ (Random) | LPIPS↓ (Policy) | LPIPS↓ (Random) |
|------|--------------|--------------|-------------|-------------|---------------|---------------|
| **World4RL** | **326.5** | **400.1** | **17.1** | 23.4 | **0.0192** | **0.0246** |
| NWM | 547.4 | 851.9 | 30.5 | 34.9 | 0.0268 | 0.0259 |
| iVideoGPT | 450.3 | 531.3 | 18.7 | 20.7 | 0.0256 | 0.0283 |
| DiWA | 803.6 | 1231.0 | 62.9 | 96.5 | 0.0804 | 0.1364 |
| DiWA (ST) | 644.8 | 880.2 | 35.1 | 52.8 | 0.0523 | 0.0596 |

> ST = Single-Task，即单任务训练评估。World4RL 在所有指标上全面领先，且不是靠模型规模取胜（330M 并非最大）。DiWA 在 multi-task 场景下表现极差，甚至会出现跨任务场景混淆。

![图 2：世界模型预测可视化对比](arXiv-2509.19080v2/fig/WorldModel_Compare.jpg)

*图 2：Coffee-Pull-v2 任务上各模型自回归预测 rollout 的视觉对比，行=模型，列=时间步（从左到右推进）。*

*第一行 Ground Truth（GT）展示的是一次**失败执行**——机械臂未能成功将咖啡杯拉到目标位置。注意 GT 的关键视觉信号：杯子的空间位置保持在被推离目标区的状态，机械臂末端执行器的轨迹与杯口无有效接触。这一定性结果说明 GT 选择了失败案例作为测试集，是对世界模型"忠实性"的严格检验——成功预测容易，忠实还原失败才难。*

*第二行 World4RL（Ours）的预测与 GT 高度一致：咖啡杯位置、机械臂姿态、背景纹理细节均被准确还原；帧间过渡平滑无跳变，自回归 50 步后仍不退化。关键信号：杯子始终未被拉到目标位置，说明 World4RL 没有"猜测"成功结果，而是忠实复现了失败动态。*

*第三行 NWM 的预测在初期帧（1-10 步）还算接近 GT，但中后期开始出现模糊伪影，且机械臂轨迹逐渐偏离——到后期帧，画面显示杯子被错误地移动到了接近目标位置的状态，即 NWM 错误地"修正"了失败轨迹为成功。类似问题在第四行 iVideoGPT 中同样存在：虽然单帧清晰度尚可，但长时自回归后时间一致性崩溃，后期帧与 GT 严重偏离。*

*第五行 DiWA（RSSM-based）的问题最为严重：画面极度模糊，不仅机械臂形状不可辨认，甚至场景中物体布局也与 GT 不一致（DiWA 在多任务训练下有时会混淆不同任务的场景背景）。单任务 DiWA（ST）虽有所改善但差距仍然很大。*

*总结：这幅对比图直接验证了论文的核心主张——扩散架构（World4RL）在长时自回归生成中的时空一致性远优于 RSSM（DiWA）、DiT（NWM）和 autoregressive transformer（iVideoGPT），且能忠实地建模失败动态而非"猜测"成功。这对 RL 训练至关重要：如果世界模型在策略犯错时生成虚假的成功画面，策略将收到错误的正面反馈，RL 优化将完全失效。*

### 4.3 策略学习效果

**表 2：Meta-World 成功率（%，3 种子平均）**

| 任务 | BC (Base) | DP | TD3+BC | IQL | IRASim | DiWA | TD-MPC2 | **World4RL** |
|------|-----------|-----|--------|-----|--------|------|---------|-------------|
| coffee-pull-v2 | 47±7 | 34±7 | 57±13 | 47±9 | 55±6 | 49±10 | 60±7 | **68±5** ↑21 |
| soccer-v2 | 18±2 | 19±4 | 22±8 | 14±7 | 28±5 | 21±6 | 24±5 | **31±4** ↑13 |
| hammer-v2 | 79±5 | 15±6 | 89±4 | 73±10 | 82±5 | 83±7 | 85±5 | **91±4** ↑12 |
| door-lock-v2 | 74±5 | 86±8 | 82±6 | 69±11 | 78±5 | 88±7 | 85±4 | **92±5** ↑14 |
| lever-pull-v2 | 31±5 | 49±5 | 39±11 | 24±3 | 33±5 | 50±9 | 39±6 | **52±8** ↑21 |
| handle-pull-v2 | 60±6 | 67±11 | 57±6 | 25±6 | 66±6 | 68±6 | 67±5 | **71±4** ↑11 |
| **平均** | **51.5** | 45.0 | 57.7 | 42.0 | 57.0 | 59.8 | 60.0 | **67.5** ↑16 |

**关键发现**：
- World4RL 在所有 6 个任务上均取得最优，平均提升 **16 个百分点**
- 在困难任务（coffee-pull、soccer、lever-pull）上提升尤其显著（11-21%），这些任务对单纯模仿学习最具挑战
- 离线 RL 方法（TD3+BC、IQL）受限于固定数据集，提升有限
- 基于世界模型的方法中，IRASim 需要测试时规划（推理成本高达 40×），DiWA 受 RSSM 架构限制

![图 3：在线样本效率对比](arXiv-2509.19080v2/fig/average_data_linear_horizontal_expert_rollout_online_scaled10.jpg)

*图 3：在线样本效率对比——水平堆积条形图，展示不同方法达到特定成功率所需的总数据量（单位为 10k steps）。*

*图表结构：横轴为 Total Samples（×10k），纵轴从上到下排列三种方法：World4RL（Ours）、RLPD、Uni-O4。每个方法的条形由不同颜色的分段堆叠组成，分别代表不同的数据来源。*

*World4RL（最上方，最短的条）：总数据量约 20k steps（即 200 条轨迹 × 50 步 ≈ 10k，加 BC rollout 等，标度约 2 个单位）。条形仅由离线数据分段组成：Expert Demonstrations（深色段，50 条 × 50 步 ≈ 2.5k）和 Policy Rollouts（浅色段，150 条 × 50 步 ≈ 7.5k），加上少量 Random Rollouts。关键特征：条形完全没有"Online Interaction"分段——World4RL 达到 ~67.5% 平均成功率所需在线交互为 **0 步**。*

*RLPD（中间）：总数据量约 54.6 个单位（546k steps）。条形由两部分组成：前半段为离线数据（Expert + Rollout，与 World4RL 相同的 10k），后半段（约 34.6 个单位）为 Online Interaction（浅色或不同填充的分段）。这意味着 RLPD 需要额外 **346k 在线交互步**才能达到 World4RL 同等性能——在线交互量是离线数据量的 17 倍以上。*

*Uni-O4（最下方，最长的条）：总数据量约 67 个单位（670k steps）。与 RLPD 类似，离线基础数据相同（10k），但需要额外约 47 个单位的在线交互（**~470k steps**）才能追平 World4RL。*

*核心信息：World4RL 的条形长度仅为其下方两个方法的 1/27 到 1/34。这个对比量级不是 20%-30% 的效率提升，而是**数量级的差异**——World4RL 用零在线交互实现了需要数十万次真机交互才能达到的性能。对于机器人操作中每次交互都涉及物理 setup、安全监督和硬件磨损的场景，这个样本效率差距意味着从"实验上不可行"到"实际可部署"的跨越。*

### 4.4 真机实验

![图 4：真机任务设置](arXiv-2509.19080v2/fig/open_drawer_compressed.jpg)

*图 4：六个真实世界操作任务的实验场景设置，使用 Franka Emika Panda 7 自由度机械臂，配备平行夹爪和 RealSense 深度相机（固定于机械臂上方提供第三视角观测）。所有任务采用 HIL-SERL 协议通过 Space Mouse 遥操作采集数据，初始场景配置和机器人起始位姿在训练/评估过程中保持固定。以下逐任务分析各场景的物理配置和操作难点：*

<!-- 6个子图，3行2列布局 -->
<table>
<tr>
<td width="50%"><img src="arXiv-2509.19080v2/fig/open_drawer_compressed.jpg" width="100%"><br><em>(a) Open Drawer — 打开抽屉</em></td>
<td width="50%"><img src="arXiv-2509.19080v2/fig/close_drawer_compressed.jpg" width="100%"><br><em>(b) Close Drawer — 关闭抽屉</em></td>
</tr>
<tr>
<td width="50%"><img src="arXiv-2509.19080v2/fig/pick_bread_in_compressed.jpg" width="100%"><br><em>(c) Pick Bread In — 将面包放入容器</em></td>
<td width="50%"><img src="arXiv-2509.19080v2/fig/pick_bread_out_compressed.jpg" width="100%"><br><em>(d) Pick Bread Out — 将面包从容器取出</em></td>
</tr>
<tr>
<td width="50%"><img src="arXiv-2509.19080v2/fig/pick_apple_compressed.jpg" width="100%"><br><em>(e) Pick Apple — 抓取苹果</em></td>
<td width="50%"><img src="arXiv-2509.19080v2/fig/press_button_compressed.jpg" width="100%"><br><em>(f) Press Button — 按下按钮</em></td>
</tr>
</table>

*子图 (a) Open Drawer：桌面放置一个标准抽屉柜，目标是用夹爪勾住抽屉把手并向外拉出。主要难点在于把手定位精度——夹爪需精准对齐窄小的把手边缘，稍有偏差就会滑脱。World4RL 达到 19/20（95%），BC 仅 12/20（60%）。*

*子图 (b) Close Drawer：与 (a) 互逆任务，需将已拉开的抽屉推回关闭位置。难点在于力的方向控制——推的方向必须与抽屉轨道方向平行，否则会卡住。World4RL 达到 20/20（100%），BC 为 15/20（75%）。*

*子图 (c) Pick Bread In：桌面有一片面包和一个开口容器，需要将面包从桌面抓取并放入容器中。这是 6 个任务中 World4RL 唯一未达最优的任务（16/20，低于 DP 的 18/20）。难点在于面包形状不规则、质地柔软，抓取策略需要适应形变，DP 的动作分块（chunking）策略在此场景下有优势。*

*子图 (d) Pick Bread Out：与 (c) 互逆，需从容器中取出面包。World4RL 达 20/20（100%），BC 仅 13/20（65%）。取出比放入对 RL 更友好——取出时面包已经位于容器内，视觉遮挡少，夹爪的插入和抓取位置更容易确定。*

*子图 (e) Pick Apple：桌面有一个苹果，需要稳定抓取并抬起。World4RL 达 19/20（95%），BC 仅 8/20（40%）。BC 在此任务上的极低成功率反映了纯模仿学习对物体形状泛化的脆弱性——苹果的曲面导致固定的抓取姿态在轻微初始位置变化时就会失败。World4RL 通过世界模型内的 RL 学会了适应物体位置的微小变化。*

*子图 (f) Press Button：桌面有一个按钮装置，目标是用夹爪末端按下按钮。World4RL 达 18/20（90%），BC 为 12/20（60%）。难点在于接触精度——按钮面积小，需要精确的末端位置控制，且按压力度不能过大（否则会推倒装置）。World4RL 优化后的策略执行更果断，减少了 BC 策略在按钮上方犹豫徘徊的行为。*

**表 3：真机成功率（20 次测试/任务）**

| 任务 | BC (Base) | DP | **World4RL** |
|------|-----------|-----|-------------|
| Pick bread out | 13/20 | 19/20 | **20/20** |
| Pick apple | 8/20 | 15/20 | **19/20** |
| Press button | 12/20 | 16/20 | **18/20** |
| Put bread in | 12/20 | **18/20** | 16/20 |
| Open drawer | 12/20 | 18/20 | **19/20** |
| Close drawer | 15/20 | **20/20** | **20/20** |
| **平均 SR** | **68.3%** | 88.3% | **93.3%** ↑25 |

**关键发现**：
- 真机平均提升 **25 个百分点**（68.3% → 93.3%），比仿真实验的 16% 提升更大
- 除 Put bread in 外，World4RL 在所有任务上均达最优或并列最优
- 观察到的行为改进：World4RL 优化后的策略执行更**果断**——抓取和放置动作快速准确，而 BC 和 DP 策略常出现犹豫或停留在中间状态不完成任务的倾向

### 4.5 消融实验

**动作编码策略对比**

**表 4：不同动作编码在视频预测上的对比**

| 编码方式 | FVD↓ (Policy) | FVD↓ (Random) | FID↓ (Policy) | FID↓ (Random) | LPIPS↓ (Policy) | LPIPS↓ (Random) |
|----------|--------------|--------------|-------------|-------------|---------------|---------------|
| **Two-hot** | **326.5** | **400.1** | **17.07** | **23.43** | **0.0192** | **0.0246** |
| One-hot | 350.3 | 471.5 | 18.52 | 26.24 | 0.0193 | 0.0257 |
| Linear | 353.4 | 514.0 | 17.85 | 23.83 | 0.0218 | 0.0250 |
| FAST | 407.0 | 748.0 | 28.92 | 36.52 | 0.0284 | 0.0409 |
| VQ-VAE | 525.6 | 860.0 | 28.60 | 43.25 | 0.0506 | 0.0633 |

> Two-hot 在所有指标上最优。One-hot 和 Linear 效果接近，但均不如 two-hot 的连续-离散混合表示。FAST 和 VQ-VAE 因有损重建导致性能显著下降。

**策略优化设计消融**

![图 5：策略优化消融实验](arXiv-2509.19080v2/fig/exp_3abl.jpg)

*图 5：策略优化消融实验，对比完整方法（Full）与两个消融变体在 Door-lock-v2（左）和 Lever-pull-v2（右）上的训练曲线。横轴为训练 Epoch，纵轴为 Success Rate。三条曲线分别对应：Full Method（实线）、w/o action std clipping（虚线）、w/o random rollouts（点划线）。阴影区域表示多个随机种子下的标准差/方差。*

*左图 Door-lock-v2：**Full Method**（实线）从 BC 初始化的 ~74% 出发，在约 25 epochs 后开始稳步攀升，约 75 epochs 附近达到平台期 ~92%，方差较小（阴影窄），训练过程平滑。**w/o action std clipping**（虚线）的曲线在整个训练过程中剧烈振荡——成功率在 50% 到 95% 之间大幅波动，说明取消标准差约束后策略频繁采样 OOD 动作，世界模型生成的轨迹质量不稳定，导致 PPO 的优势估计方差极大。最终收敛在约 78%，比 Full 低 14 个百分点。**w/o random rollouts**（点划线）曲线相对稳定但始终低于 Full——上升期斜率较平缓，约 50 epochs 后停滞在 ~78%。这说明缺乏随机 rollout 数据的世界模型在策略探索超出 BC 分布时无法提供可靠的转移预测，形成了事实上的性能天花板。*

*右图 Lever-pull-v2：起始成功率更低（~31%），任务难度更大。**Full Method**（实线）在约 50 epochs 前缓慢爬升，此后加速收敛至 ~52%，但方差比 Door-lock-v2 大（阴影更宽），说明 Lever-pull-v2 本身具有更高的随机性。**w/o action std clipping**（虚线）的表现最为惨烈——曲线长时间在 20%-40% 之间剧烈波动，甚至出现多个回退到接近 0% 的崩溃点，最终约 35%，比 Full 低 17 个百分点，方差接近 Full 的 3 倍。**w/o random rollouts**（点划线）前期爬升速度与 Full 接近，但约 30 epochs 后就停滞在 ~38%，与 Full 的差距持续拉大——这表明随着策略不断优化并逐渐偏离 BC 初始分布，没有随机数据支持的世界模型越来越无法提供有效训练信号。*

*跨任务对比：Door-lock-v2 的成功率上限更高（~92% vs ~52%），且两种消融的绝对下降幅度接近（均为 ~14 个百分点），但相对比例不同（14/92 ≈ 15% vs 17/52 ≈ 33%），说明在更难的任务上，两项设计选择的重要性不成比例地增大。此外，w/o action std clipping 在两项任务上都导致方差显著增大，而 w/o random rollouts 主要影响最终收敛值而非训练稳定性——这揭示了两个设计的互补关系：**受控探索（std clipping）保证训练稳定性，扩大动作覆盖（random rollouts）提升性能上限**。*

---

## 五、关键洞察与技术亮点

1. **"先想再做"优于"纯反应"**：即使只用 1 个 latent token 进行"思考"也优于纯 action-only，验证了内化世界模型进行推理的核心价值。

2. **扩散模型的时空调和性**是比模型规模更关键的优势：World4RL（330M）在 FVD/FID/LPIPS 上全面超越更大的 iVideoGPT（430M），本质在于扩散架构对 sharp、temporally coherent 生成的天然优势。

3. **失败建模与成功建模同样重要**：可视化实验证明，World4RL 能忠实还原失败轨迹，而 NWM/iVideoGPT 会错误生成成功结果——这在 RL 中是不可接受的，因为策略需要从失败中学习。

4. **Two-hot 编码作为"桥梁"**：无损失连续表示连接了连续策略输出和离散 bin 结构，避免了有损重建带来的梯度不稳定，是实现端到端优化的关键设计。

5. **受控探索 = OOD 防护**：$\sigma \le e^0$ 看似简单的技巧，实质上是对"世界模型信任域"的工程约束——策略只探索世界模型"见过"的动作空间。

6. **数据混合策略**：专家+策略+随机三方数据构成了世界模型训练的完备覆盖——高质量、策略相关、边界探索三者缺一不可。

7. **离线胜过在线**：仅用固定数据集的世界模型内 RL 优化，效果优于需要 300k-470k 在线交互步的 offline-to-online 方法，样本效率优势显著。

---

## 六、代码实现解读

未找到公开代码仓库，以下基于论文内容推断核心模块的实现要点。

### 6.1 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    World4RL 系统架构                       │
├─────────────────────────────────────────────────────────┤
│  Stage 1: Pre-training                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ BC Policy│  │ Reward Class │  │ Diffusion Trans   │  │
│  │ π_ξ(x_t) │  │ C_ψ(x_{t+1}) │  │ D_θ(x^τ; c, z)   │  │
│  │ → a_t    │  │ → r ∈ {0,1}  │  │ → x_{t+1}         │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
│       │               │                   │              │
│  D_exp only    D_exp + D_roll    D_exp + D_roll + D_rand│
├─────────────────────────────────────────────────────────┤
│  Stage 2: Policy Optimization (world model FROZEN)      │
│                                                          │
│  x_t ──→ [π_ξ] ──→ a_t ──→ [TwoHot] ──→ z_t            │
│    │                                          │          │
│    └──── x_{t-T:t}, z_{t-T:t} ──→ [D_θ] ──→ x̃_{t+1}     │
│                                        │                │
│                                        └──→ [C_ψ] ──→ r │
│                                                   │      │
│                              (x_t, a_t, r, x̃_{t+1})      │
│                                        │                │
│                                   [PPO Update]          │
└─────────────────────────────────────────────────────────┘
```

### 6.2 核心模块推演

#### Diffusion Transition Model（U-Net 2D, ~330M 参数）

```
输入：
  - x^τ_{t+1}: 噪声化的目标帧（扩散过程中的中间状态）
  - x⁰_{t-T:t}: 历史T帧原始观测 (conditioning)
  - z_{t-T:t}: 历史T步的two-hot编码动作 (conditioning)

输出：预测的去噪目标 x̃_{t+1}

推理时自回归生成：
  初始帧 → [采样 a_0] → [生成 x̃_1] → [采样 a_1] → [生成 x̃_2] → ...
```

关键实现细节：
- 网络骨干 $F_\theta$ 使用 U-Net 2D 架构
- 噪声调度和超参数（$c_{in}, c_{out}$ 等）遵循 EDM 设计
- 历史帧和编码动作通过条件注入 U-Net（类似 ControlNet 范式）

#### Two-Hot 动作编码

```
def twohot_encode(action, bins, K=21):
    """将连续动作编码为 K 维 two-hot 向量"""
    # action: [B, act_dim], bins: [K]
    idx = searchsorted(bins, action)  # 找到最近的 bin 索引
    # 计算到两个最近 bin 的插值权重
    lower = bins[idx]
    upper = bins[idx+1]
    w_upper = (action - lower) / (upper - lower)
    w_lower = 1 - w_upper
    # 构造 K 维稀疏向量
    vec = zeros(B, act_dim, K)
    vec[..., idx] = w_lower
    vec[..., idx+1] = w_upper
    return vec
```

#### PPO 训练循环（受控探索）

```
for epoch in range(max_epochs):
    # 1. 世界模型内 rollout
    obs = env.reset()
    for t in range(horizon):
        mu, sigma = policy(obs)
        sigma = clamp(sigma, max=1.0)  # σ ≤ e⁰, 即 clamp to 1.0
        action = Normal(mu, sigma).sample()
        z = twohot_encode(action)
        next_obs = diffusion_model(obs_history, z_history)
        reward = reward_classifier(next_obs)
        buffer.add(obs, action, reward, next_obs)

    # 2. PPO 更新
    if len(buffer) >= ppo_batch_size:
        ppo_update(policy, value_net, buffer)
        buffer.clear()
```

### 6.3 训练资源

- GPU：NVIDIA A800 40GB
- 世界模型预训练：4 × A800，约 20 小时
- 单任务策略优化：1 × A800，约 6 小时

---

## 七、局限性

1. **视觉分辨率与模型容量受限**：受计算资源约束，当前实现使用中等分辨率和模型规模，可能限制了想象 rollouts 的保真度上限。更高分辨率可能进一步提升世界建模质量。

2. **动作分布约束**：世界模型的建模能力最终受限于离线数据集中的动作分布。即使有随机 rollout 和受控探索，策略在 RL 过程中仍可能超出世界模型的有效泛化范围。

3. **单一真机平台验证**：真机实验仅在 Franka Emika Panda 上进行，未验证跨机器人平台（如不同机械臂构型）的迁移能力。

4. **任务类型有限**：实验任务均为相对短视距（50 步/条）的桌面操作，未测试长视距任务或移动操作场景。

5. **世界模型冻结策略**：Stage 2 中世界模型完全冻结，无法利用 RL 过程中收集的新数据来改进动态模型——这在某些场景下限制了策略的上限。

---

## 八、关键概念速查

| 概念 | 解释 |
|------|------|
| **Diffusion World Model** | 用扩散模型学习环境动态 $P(x_{t+1}|x_t, a_t)$，替代传统仿真器 |
| **Two-Hot Encoding** | 连续动作的双 bin 插值编码，无损失、可微分，$K=21$ bins |
| **EDM Preconditioning** | 扩散模型的预条件去噪公式，通过 $c_{skip}, c_{out}, c_{in}, c_{noise}$ 稳定训练 |
| **Imagined Rollout** | 完全在世界模型内自回归生成的轨迹，无需真实环境 |
| **Controlled Exploration** | $\sigma \le e^0$ 的策略标准差裁剪，防止 OOD 动作 |
| **Reward Classifier** | ResNet18 二分类器，判断观测是否为成功状态 |
| **PPO** | Proximal Policy Optimization，本文使用的策略优化算法 |
| **FVD** | Fréchet Video Distance，视频级时空一致性指标 |
| **FID** | Fréchet Inception Distance，图像分布质量指标 |
| **LPIPS** | Learned Perceptual Image Patch Similarity，感知相似度指标 |
| **RSSM** | Recurrent State-Space Model，DiWA 使用的世界模型架构，生成质量差于扩散模型 |
