# RTC（Real-Time Chunking）实现原理

对应论文：Black, Galliker, Levine, *Real-Time Execution of Action Chunking Flow Policies*
（[arXiv:2506.07339](https://arxiv.org/abs/2506.07339)）；参考实现：
[real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) `src/model.py`。
本文说明 `tasl/rtc/` 里每一步对应论文哪条公式、在我们的 FR3 + pi05_droid 系统里怎么落地。

---

## 1. 要解决的问题

pi05_droid 一次推理输出一段 **action chunk** `A = (a_0, …, a_{H-1})`（`H=16`，franka LoRA 配置；base 是 15），
控制器以 `Δt = 1/15 s` 逐个执行。现在 portal 的同步循环是：

```
obs → infer(~95 ms) → 执行 a_0..a_3 (4 tick = 267 ms) → obs → infer → …
```

两个缺陷：

1. **停顿**：每段 chunk 之间机器人等推理，真实动作节奏与训练数据（连续 15 Hz）不一致。
2. **边界跳变**：新 chunk 是在不知道旧 chunk 的情况下独立采样的；flow 策略是多模态的，
   两段 chunk 可能选不同"策略"（论文图 2：绕障碍物上方 vs 下方），切换处出现大加速度。

朴素的异步（一边执行旧 chunk 一边算新 chunk，算好就切）解决了 1 但让 2 更严重：
推理需要 `d` 个 tick，切换时执行到 `a_d`，而新 chunk 的 `a'_d` 对旧 chunk 一无所知。

RTC 的核心想法：**把"生成一个与已执行前缀兼容的新 chunk"当成 inpainting 问题**——
新 chunk 的前 `d` 个动作在它可用之前就已经被执行掉了，那就把它们"冻结"成已知值，
让 flow 去噪过程在这个约束下补全其余部分。

---

## 2. Flow matching 回顾（openpi 的时间约定）

openpi 的 π0/π0.5 训练目标：

```
x_t = t·ε + (1−t)·a          ε ~ N(0, I),  t ∈ [0,1]
v_θ(x_t, o, t) ≈ u_t = ε − a
```

**t=1 是纯噪声，t=0 是干净动作**（`pi0.py` 注释："opposite of the pi0 paper, and I'm sorry"）。
采样从 `t=1` 积分到 `0`，`n=10` 步，`dt = −1/n`：

```
x_{t+dt} = x_t + dt · v_θ(x_t, o, t)
```

论文用的是相反方向：τ∈[0,1)，τ=0 噪声，τ=1 数据，`A^{τ+1/n} = A^τ + (1/n)·v_π`。
两者关系：**τ = 1 − t，v_π = −v_θ**。后面所有公式我都换算到 openpi 约定，
代码（`rtc_math.py`）直接在这个约定下工作，这样 `x + dt·v` 这一步在有/无 RTC 两条分支里完全一样。

一个关键量——**去噪估计**（论文 Eq.3 `Â¹ = A^τ + (1−τ)v_π`）：

```
â(x_t) = x_t − t · v_θ(x_t, t)        # 验证：代入 x_t 与 u_t 得 t·ε+(1−t)a − t(ε−a) = a ✓
```

它是"从当前噪声状态一步外推出的干净 chunk"，是 guidance 作用的对象。

---

## 3. 把 RTC 写成 inpainting

### 3.1 目标 Y 与权重 W（论文 Eq.5）

设上一段 chunk `A_prev`，开始新推理时已执行了 `s` 个动作（`s ≥ s_min`），
预计推理耗时 `d` 个 tick。新 chunk 的第 `i` 个动作对应旧 chunk 的第 `s+i` 个：

```
Y = [A_prev[s], A_prev[s+1], …, A_prev[H−1], 0, …, 0]     # 平移 s，右侧补零到 H
```

权重（`rtc_math.get_prefix_weights(start=d, end=H−s, total=H, schedule)`）：

```
        ┌ 1                               i < d          冻结前缀：新 chunk 可用前这些已经执行掉了
W_i =   │ c_i · (e^{c_i} − 1)/(e − 1)     d ≤ i < H−s    软掩码：越远越不确定，指数衰减到 0
        └ 0                               i ≥ H−s        旧 chunk 没有的部分，完全自由生成
其中 c_i = (H − s − i)/(H − s − d + 1)
```

四种 schedule 都实现了：`zeros`（只冻结前缀，硬掩码）、`linear`、`exp`（论文默认）、`ones`
（整个重叠区权重 1）。论文实验：软掩码优于硬掩码，`d` 越小差距越大。

补零只是占位——那些行权重为 0，值不参与任何计算。

### 3.2 ΠGDM guidance（论文 Eq.2 / Eq.4）

每一步去噪，在模型速度上加一个把 `â` 拉向 `Y`（按 W 加权）的修正项：

```
论文:   v_guided = v_π + min(β, (1−τ)/(τ·r_τ²)) · g
        g = [W ⊙ (Y − Â¹)]ᵀ · ∂Â¹/∂A^τ          ← vector-Jacobian product
        r_τ² = (1−τ)² / (τ² + (1−τ)²)
```

直觉：`Â¹` 是对最终结果的估计，`W⊙(Y−Â¹)` 是"还差多少"；通过 Jacobian 反传，
把这个差距转换成"当前噪声状态该往哪动"。`r_τ²` 来自 ΠGDM 对 `Â¹` 不确定度的高斯近似
（噪声大时估计不可靠，修正应弱），`(1−τ)/τ` 是把 score 换算成 flow 速度的因子。
`min(β, ·)` 是论文相对 ΠGDM 的新增——**只有 5–10 步去噪时不截断会发散**（附录 A.2）。

换算到 openpi 约定（τ = 1−t，v_π = −v_θ，dt = −1/n）：

```
x_{t+dt} = x_t + (1/n)(v_π + w·g) = x_t + dt·(v_θ − w·g)

w(t) = min(β, (τ² + (1−τ)²) / (τ(1−τ)))       τ = 1−t     # 合并 (1−τ)/τ 与 1/r_τ² 后的形式
g    = vjp(â, x_t)[ W ⊙ (Y − â(x_t)) ]
```

代码（`rtc_math.guided_velocity`）：

```python
def denoiser(x):                      # â(x) 及其副产物 v
    v = velocity_fn(x)
    return x - t * v, v
x0_hat, vjp_fn, v_t = jax.vjp(denoiser, x_t, has_aux=True)
error      = (Y - x0_hat) * W[None, :, None]
correction = vjp_fn(error)[0]          # g
return v_t - guidance_weight(t, beta) * correction
```

`jax.vjp` 对 `â` 关于 `x_t` 求 VJP，会自动反传穿过 action expert 的整个 suffix 前向
（16 个 action token + state token，attend 到缓存的 prefix KV）。代价 ≈ 一次额外的反向传播：
实测 95 ms → 137 ms（论文 π0.5：76 → 97 ms）。

`w(t)` 在 t=1（纯噪声）和 t→0 两端都发散，被 β 截住；中段（t=0.5）为 2。
`test_rtc_math.py` 用一个有解析解的高斯 flow 验证：冻结区被拉到 Y、软掩码区拉力单调衰减、
权重 0 的尾部与无 guidance 采样逐元素一致。

### 3.3 与 π0.5 采样循环的接合（`rtc_policy.sample_actions_rtc`）

`Pi0.sample_actions` 的结构：先对 prefix（图像 + 语言）做一次前向填 KV cache，
再 `while_loop` 做 n 步 suffix 前向。`sample_actions_rtc` 逐行复刻这个函数
（openpi commit c23745b），只把积分步里的 `velocity(x_t, time)` 换成 `guided_velocity(…)`，
`W` 在循环外算一次。`prev_action_chunk` 为 batch 内的 `[1, H, 32]`，与 `x_t` 同形。

为什么不改 openpi：用 `RTCPolicy` 包一层，`infer()` 看到请求里没有 `rtc` 键就原样转发给
被包的 `Policy.infer`，因此 vanilla 客户端（`examples/droid/main.py`、旧 portal 循环）行为不变；
有 `rtc` 键才走 guided 分支。JIT 用 `nnx.split/merge` 冻结模块状态（同 openpi `module_jit` 的做法），
`d`、`H−s`、β、schedule 编码都是 traced 标量，改参数不重新编译；启动时 `warmup()` 用
`fake_obs` 把 plain + guided 两条 trace 都编好（共 16 s）。

---

## 4. 系统层：论文 Algorithm 1 → `executor.py`

### 4.1 两个线程

```
controller 线程 (15 Hz，永不等待)              inference 线程
────────────────────────────────              ─────────────────────────────────────
每 Δt：                                        循环：
  i = t; a = A_cur[i] (i<H) 或 None(hold)        等 t ≥ s_min
  t += 1; notify                                 obs = 采集(相机 + 关节)
  发送 a（None → 零速度 + 保持夹爪）               s = t;  Y = shift(A_cur_model, s)
                                                 d = min(max(最近 b 次实测延迟), H−s)
                                                 A_new = infer(obs, Y, d, H−s)    ← 期间 controller 继续吃 A_cur
                                                 elapsed = t − s                  ← 推理期间实际过了几个 tick
                                                 A_cur = A_new;  t = elapsed      ← 切换并对齐
                                                 记录 elapsed 进延迟缓冲
```

对齐的含义：新 chunk 的第 0 个动作对应采集 obs 那一 tick（旧 chunk 的第 `s` 个）。
推理花了 `elapsed` 个 tick，这期间 controller 执行的是 `A_cur[s..s+elapsed)`——
正是 Y 的前 `elapsed` 行，而新 chunk 的前 `d ≈ elapsed` 行被 guidance 冻结成这些值，
所以从 `A_new[elapsed]` 接着执行没有跳变。若 `elapsed > d`（延迟低估），
接上的是软掩码区，仍被拉向旧 chunk，只是弱一些。

### 4.2 时序示意（H=16，s_min=4，d=3）

```
tick:   0   1   2   3 | 4   5   6 | 7   8   9  10  11 ...
执行:  a0  a1  a2  a3 | a4  a5  a6 | a'7 a'8 a'9 ...
                      ↑ t=4≥s_min: 采 obs, Y=A[4:], 开始推理(≈2.1 tick)
                                  ↑ A' 到达, elapsed=3 → t=3, 从 a'_3 继续
       新 chunk A' 的坐标:        a'0 a'1 a'2 | a'3 …
                                  └─ 冻结 = a4 a5 a6 ┘  软掩码 a'3..a'11 ≈ a7..a15   自由 a'12..a'15
```

### 4.3 约束与 FR3 参数

论文要求 `d ≤ s ≤ H − d`（新 chunk 到达前旧 chunk 不能用完；冻结区不能覆盖整段）。
我们的数字：`H = 16`，`Δt = 66.7 ms`，guided 推理 137 ms + obs 采集 ≈ 2.1–2.5 tick
→ `delay_init = 3`，`s_min = 4`，prefix horizon `H − s = 12`。

`d` 的估计是**保守的**：取最近 `delay_buffer = 5` 次实测的最大值（论文同样做法），
宁可多冻结一个 tick 也不要出现"新 chunk 到了但它的冻结区比已执行的短"。

饥饿保护：若推理比整段 chunk 还慢（`elapsed ≥ H`），controller 发零速度 hold，
inference 线程退化成 plain 采样（`s ≥ H` 时 Y 为空），不会崩溃；`starved_ticks` 计数暴露在 UI 上。

### 4.4 warm-up

第一次 guided 调用要编译 JAX trace。serve 启动时已经 `warmup()`，executor 默认再做一次
throwaway guided 推理（此时机器人静止），确认编译完成后才启动 controller。

---

## 5. 协议与模块边界

```
dashboard (client, /usr/bin/python3)                 serve (openpi venv, GPU)
────────────────────────────────────                 ────────────────────────────────
obs + obs["rtc"] = {                                 RTCPolicy.infer:
  prev_actions: Y  [H,32] 模型空间 | None    ──WS──▶    无 rtc 键 → 原 Policy.infer
  inference_delay: d                                   有 → transforms → sample_actions_rtc
  prefix_attention_horizon: H−s                        ◀── actions [H,8] (反归一化, 可执行)
  schedule, max_guidance_weight }                          actions_model [H,32] (模型空间)
                                                           rtc: {guided, d, H−s}
```

**为什么 `prev_actions` 用模型空间**：guidance 作用在归一化后的 32 维 `x_t` 上。
让 client 原样保存上一次响应的 `actions_model`、平移补零后传回，server 就完全无状态，
不需要在 server 端缓存"上一段"或做反归一化 → 归一化的往返。client 把它当不透明 blob。

**开关**：portal 的 *Load w/o RTC* 起 openpi 原版 `scripts/serve_policy.py`，
*Load with RTC* 起 `tasl/rtc/scripts/serve_policy.py`（同 CLI）。`ServeManager.rtc` 记住
加载的是哪种，eval 循环据此选同步路径或 RTC 路径（`dashboard_hook.active`）。
Dashboard 重启时从 serve 进程命令行恢复这个标记。

---

## 6. 已验证 / 未验证

已验证（2026-08-25）：
- 单元测试 17 个（数学、executor 时序、dashboard 胶水）。
- 真 ckpt（`10task/16000`）离线冒烟：guided 时新 chunk 前 d 个动作与旧 chunk 对应动作的
  L1 误差 **0.040**，独立 plain 采样为 0.095，硬掩码 0.026；`ones > exp > zeros` 对软掩码区
  的拉力顺序正确；尾部不受影响。延迟 plain 91 ms / guided 137 ms。

未验证：**真机**。首次真机建议 `s_min=4, delay_init=3, β=5, exp`，同一任务各跑一条
sync / RTC 对比，看 `rtc.json` 里 `delays` 与 `starved_ticks`；若 `delays` 常到 4，把 `delay_init` 调到 4。

局限：guidance 是近似 inpainting（冻结区误差不为 0，所以论文强调软掩码）；
hooked PyTorch serve（steer 实验）没有 RTC；legacy 二值夹爪模式不在 RTC 路径里。

---

## 7. 与官方实现的逐项对照（real-time-chunking-kinetix @ 9296f31）

官方 repo 是论文的 **Kinetix 仿真**代码：`src/model.py` 里 `get_prefix_weights` / `FlowPolicy.realtime_action`
是 guided 采样器，`src/eval_flow.py` 用固定的 `inference_delay` / `execute_horizon` 模拟异步执行。
真机的 Algorithm 1（线程、延迟估计）官方没有开源（论文 checklist 注明 runtime 代码为私有）。

| 项 | 官方 (`model.py`) | 本实现 (`rtc_math.py` / `rtc_policy.py`) | 结论 |
|---|---|---|---|
| 时间约定 | τ: 0=噪声 → 1=数据，`dt=+1/n`，在 τ=0,1/n,…,(n−1)/n 求值 | t: 1=噪声 → 0=数据（openpi），`dt=−1/n`，在 t=1,…,1/n 求值 | 同一网格（τ=1−t） |
| 去噪估计 | `x_1 = x_t + v·(1−t)` | `x0_hat = x − t·v` | 等价（v_openpi = −v_paper） |
| 误差项 | `(y − x_1) * weights[:, None]` | `(Y − x0_hat) * W[None,:,None]` | 相同 |
| VJP | `jax.vjp(denoiser, x_t, has_aux=True)`，`vmap` 逐样本 | 同，对整个 batch 一次做（π0 样本间独立，batch=1） | 相同 |
| guidance 权重 | `min(nan_to_num((1−t)/t, posinf=β)·(t²+(1−t)²)/(1−t)², β)` | `min((τ²+(1−τ)²)/max(τ(1−τ),1e-6), β)`，τ=1−t | 代数相同；τ=0 两者都截到 β |
| 速度更新 | `v + w·g`，再 `x + dt·v` | `v − w·g`，再 `x + dt·v`（dt<0） | 等价（符号随约定翻转） |
| `get_prefix_weights` | 字符串分支 | 整数码 + `jnp.select`（可 trace），冻结区强制精确为 1 | 数值相同（exp 调度差 1 ulp） |
| 默认超参 | `exp`，β=5 | `exp`，β=5 | 相同 |
| 去噪步数 | Kinetix 5 步；论文真机 π0.5 也是 n=5 | 沿用 openpi pi05_droid 的 10 步 | **有意差异**：可通过 serve 的 `sample_kwargs` 改为 5（延迟约减半） |
| `simulated_delay` 分支 | 训练时 RTC（第二篇论文） | 未实现 | 不在范围 |

执行循环（官方 `eval_flow.py` vs `executor.py`）：

| 项 | 官方仿真 | 本实现 |
|---|---|---|
| prev chunk 的构造 | 上一段生成后立刻 `chunk[s:]` + 补零（`execute_horizon` 平移） | 推理开始时 `shift(A_cur_model, s)`，s = 此刻已消费数 | 相同 |
| prefix horizon | `H − execute_horizon` | `H − s` | 相同 |
| 执行的动作 | `prev[:d]` 然后 `new[d:s]` | controller 一直吃旧 chunk 直到新 chunk 到达（`elapsed` tick），再从 `new[elapsed:]` 接 | `elapsed == d` 时逐动作相同；`elapsed ≠ d` 是真机才有的情形，按论文 Algorithm 1 处理 |
| d、s | 固定（用于 sweep） | d = 最近 b 次实测延迟的 max，s = max(d_prev, s_min)（Algorithm 1 第 13–17 行） | 官方无对应代码，按论文 |
| 首段 chunk | 用 reset obs 生成，未平移地作为 prev（此时什么都没执行） | 先 plain 生成，执行 s_min 个后再平移 | 都自洽 |

数值等价测试 `tests/test_rtc_vs_official.py`：把官方 `get_prefix_weights` 与 `realtime_action` 的 ΠGDM 步
**逐字复制**（论文时间约定），与本实现用同一 toy flow 场、同一噪声各跑一遍——
4 种 schedule × 4 组 (d, s) 的输出逐元素一致（rtol 1e-5），7 组 (start, end, total) 的权重一致。

### 7.1 与 LeRobot / Intel pi05-rtc-ov 的对照

Intel 的 [pi05_with_rtc](https://docs.openedgeplatform.intel.com/2026.0/edge-ai-suites/robotics-ai-suite/embodied/sample_pipelines/pi05_with_rtc.html)
管线 = **LeRobot 上游的 RTC 实现**（`src/lerobot/policies/rtc/`，@ bf31dd7）+ 9 个 OpenVINO 导出补丁
（`edge-ai-suites/robotics-ai-suite/pipelines/pi05-rtc-ov/patches/`）。

**guided 采样（LeRobot `RTCProcessor.denoise_step`）与本实现逐项相同**——它也是从 openpi 移植的，
所以时间约定、去噪估计、修正符号和我推导出的完全一致（它的注释原话："In the original implementation, the
time goes from 0 to 1 and in our implementation, the time goes from 1 to 0, so we need to invert the time"）：

| | LeRobot `modeling_rtc.py` | 本实现 |
|---|---|---|
| 时间 | `tau = 1 - time`（openpi t） | 同 |
| 去噪估计 | `x1_t = x_t - time * v_t` | `x - t·v` |
| 误差 / VJP | `err = (prev - x1_t) * weights`；`torch.autograd.grad(x1_t, x_t, err)` | 同；`jax.vjp` |
| 权重 | 同 Eq.4 公式（`inv_r2`, `nan_to_num`, `minimum(·, β)`） | 同 |
| 更新 | `v_t - guidance_weight * correction` | 同 |
| prefix 权重 | 前缀 ones + `linspace` 内插 + 尾部 zeros；exp 同式 | 数值相同 |
| 首段 chunk | `prev_chunk_left_over=None` → 不加 guidance | 同 |
| 去噪步数 | `num_inference_steps`（π0.5 默认 10） | 同（10） |
| **默认超参** | LINEAR，β=10，`execution_horizon=10` | exp，β=5（论文/官方默认） |

数值测试 `tests/test_rtc_vs_lerobot.py`：LeRobot 的 guidance 与 prefix 权重代码逐字移植成 torch，
与本实现同噪声同 toy 场对比——权重（4 调度 × 6 组）与采样结果（2 调度 × 3 组 d/end/β）逐元素一致（rtol 1e-4）。

命名差异要注意：LeRobot 的 `execution_horizon` 传给 `get_prefix_weights` 的是 **`end`**，即论文的
prefix attention horizon `H − s`，不是论文的执行 horizon `s`。Intel 的 `--rtc_horizon`（默认 45，chunk 50，50 Hz）
也是这个量：队列剩余 ≤ 45 时触发推理（即消费了 s = 5 个动作 ≈ 100 ms），attention 覆盖剩下的 45 个。

**异步执行**（LeRobot `rollout/inference/rtc.py`、Intel `eval_aloha.py`）与本 executor 的对照：

| | LeRobot / Intel | 本实现 |
|---|---|---|
| 触发 | 队列剩余 ≤ 阈值 | 已消费 `t ≥ s_min`（等价：剩余 ≤ H − s_min） |
| prev chunk | `queue.get_left_over()` = 未消费部分（已平移），在 denoise 里补零 | `shift(A_cur_model, s)`，补零 |
| d 的估计 | `ceil(max_latency / Δt)`，latency tracker 取窗口 max | 最近 b 次实测 tick 数的 max（同思路，直接数 tick 而不是除 Δt） |
| 切换对齐 | `merge()` 用新 chunk **替换**队列并丢掉前 `real_delay` 个（`real_delay = ceil(总耗时/Δt)`，并与实际消费数 `indexes_diff` 比对、不一致只记日志） | `t = elapsed`（= `indexes_diff`，直接用实际消费数） |
| 饥饿 | Intel：队列空了则 `real_delay = 推理前队列长度` | controller 发 hold，`t` 继续累加 |
| 首段 | 无 guidance | 同 |

结论：**算法核心三者一致（本实现 ≡ 官方 kinetix ≡ LeRobot）**；差异只在默认超参和工程细节。

**Intel OpenVINO 导出版是个例外**：`convert_ov_rtc.py::rtc_denoise_step`（patch 0002，0004 保留）里
`correction = err`——因为 autograd 无法导出进 OpenVINO 图，它把 VJP 去掉了，等价于假设 ∂x̂/∂x_t = I，
即修正项直接用 `W ⊙ (Y − x̂)` 而不是它经雅可比反传后的量。这不再是论文的 ΠGDM guidance，是一个
一阶近似（省掉了反向传播，推理成本 ≈ 无 guidance）。toy 场上同噪声对比：冻结区误差 论文/VJP 0.117 vs
Intel 0.224（`test_intel_openvino_variant_is_a_different_algorithm`）。另外它的 `get_prefix_weights`
分支写反了（LINEAR 会 raise、其它非 EXP 调度当 linear 用），但默认 EXP 不受影响。
如果以后想要"零反向传播"的低延迟版本，这个近似可以作为一个可选项加进 `rtc_math`，但要按实验对比后再用。
