# 真机 rollout 操作手册：T-Rex

在 OpenArm + Sharpa Wave 左手真机上跑 `zhx-tactile-steering/T-Rex-openarm-sharpa-left-egg`。

**这份是"站在机器人前面照着敲"的操作单**，只写和 pi0.5 那份
（`~/Public/openpi/RUNBOOK.md`）**不一样**的地方；臂的 bring-up、走 home、preflight、起 teleop
节点、收机这些步骤和 pi0.5 完全相同，本文只标 "同 pi0.5 §N"，不重抄。为什么这么设计、
动作怎么解码、触觉怎么对齐，在同目录的 [`README.md`](README.md)。

**每一步都给了判据。判据不过就停下，别往下走。** 标 ⚠️ 的步骤真机会动。

---

## 0 · 和 pi0.5 的差别，先看这个

| | pi0.5 | T-Rex |
|---|---|---|
| server | `openpi/.venv`，`sharpa_serve.sh` | **`T-Rex/.venv`**，`scripts/serve_openarm.sh` |
| 客户端 | `cli.py --prompt "pick up the egg"` | `cli.py --backend trex --prompt "Pick up the egg with your left hand and place it down."` |
| 触觉 | 不用 | **必须**：`sharpa_tap.py`（sudo）+ Pilot 开着且触觉视图打开 |
| 相机 | 短边 256 | 头 640×400、腕 640×360（`--backend trex` 自动） |
| chunk | 30 步 | **16 步** → `--infer-lead` 最大 7，`--chunk-steps` 最小 9 |
| 手指包络 | `--hand-norm-stats norm_stats.json` | `--trex-stats <ckpt>/stats_data.json` |
| 起跑前 | — | **触觉校准 2 s**：手悬空、什么都不碰 |
| 终端数 | 6 | **7**（多一个 tap） |

### 终端分配

| # | 干什么 | 谁的解释器 |
|---|---|---|
| 1 | policy server | `T-Rex/.venv`（`serve_openarm.sh` 自己处理） |
| 2 | `ping_trex.py` / 临时查东西 | `T-Rex/.venv` |
| 3 | 手：`probe_hand_units.py` | rig venv |
| 4 | 臂 bring-up，**起了就不要碰** | ROS |
| 5 | teleop 节点 | rig venv + ROS |
| 6 | `cli.py`，rollout 客户端 | rig venv + ROS |
| 7 | **`sharpa_tap.py`**，触觉 tap | sudo，系统 python |

每开一个新终端先跑：

```bash
export V=/home/yiming/openarm/openarm_track/vr          # rig checkout（分支 trex-backend）
export TREX=/home/yiming/Public/T-Rex                    # 分支 rollout/openarm-sharpa-left
export CKPT=$TREX/checkpoints/openarm-sharpa-left-egg/checkpoint-29-49290
```

终端 4、5、6 还要 ROS 的三行（同 pi0.5 §0）。

---

## 1 · 起 policy server（不动机器人，可以提前很久做）

第一次先建 venv（笔记本是 RTX 5090，`pyproject.toml` 钉的 torch 2.6/cu124 没有 sm_120 的核，
所以不能 `uv sync`）：

```bash
cd $TREX && scripts/setup_rollout_env.sh        # 一次性；网慢会超时，重跑会续
```

**判据**：最后打出 `device NVIDIA GeForce RTX 5090 Laptop GPU sm_120` 和 `bf16 matmul ok: True`。

checkpoint 要五样东西齐：`config.json  model.pt(8.5 GB)  processor/  stats_data.json  training_args.json`。
**数文件**：`hf download --include` 匹配不到时退出码 0、一个字节不下。

```bash
# 终端 1
cd $TREX
scripts/serve_openarm.sh $CKPT
```

**判据**：日志里要有这几行——

```
Auto-detected tactile_intermediate_size=1536 from training_args.json     （以及 use_tactile_code / vqvae / cascaded 几行）
Checkpoint loaded: missing=0, unexpected=0
Embedded VQ-VAE encodes in torch.bfloat16 (training used torch.bfloat16)
warm-up slow_and_fast: NNN ms
serving on ws://0.0.0.0:8000  (metadata: {'model': 'trex', 'horizon': 16, 'action_dim': 62, ...})
```

- `missing` 不是 0 → 这个目录不是完整的 checkpoint，或者 `--action_dim` 被人改了。停。
- `Embedded VQ-VAE encodes in torch.float32` → 有人传了 `--vqvae_fp32 1`。训练是 bf16，
  fp32 会让 14 % 的触觉码变掉。去掉。
- `--image_size 384 288` 由 `serve_openarm.sh` 固定；这一项 **不在** `training_args.json` 里，
  传别的值不报错、只是喂错分辨率。

```bash
# 终端 2
cd $TREX
.venv/bin/python hardware_code/openarm/ping_trex.py --n 30 --stats $CKPT/stats_data.json
```

**判据**：`OK -- p95 fits in 233 ms`，chunk `(16, 62)`，没有 `CHUNK PROBLEMS`。它最后一行直接给出
该传给 `cli.py` 的 `--infer-lead N --chunk-steps M`，**记下来**。

超预算的话它会说明是"换个 lead 能放下"还是"任何 lead 都放不下"；后者按 `README.md` 的
Latency 一节处理（先降 `--cascaded_total_steps`），不要硬跑——loop 会在每个 chunk 边界卡住。

> 权重还没下完也能先测时延：`scripts/serve_openarm.sh $CKPT --random_weights 1`。
> 输出是垃圾，时间是真的；客户端看到 `random_weights` 会拒绝 `--enable-*`。

---

## 2 · 触觉 tap 与 Pilot（pi0.5 没有这一节）

```bash
# 终端 7
cd $V && sudo ./collect/sharpa_tap.py --serve /tmp/sharpa_tap.sock --stats
```

然后开 Pilot 并**打开触觉/形变视图**：

```bash
/opt/sharpa-pilot/sharpa-pilot --no-sandbox
```

设备只在有客户端请求时才算形变图，tap 只能看到 Pilot 请求了什么——视图关着，deform 就全是平面，
而且**盘上和日志里都不报错**。采集时是这么干的（`vr/collect/README.md` §1.4），rollout 也一样。

**判据**：终端 7 的 `--stats` 每秒打出 5 个通道都在 ~30 Hz 收帧。之后第 6 步 `cli.py` 起来时会打
`tactile: N frames off the tap, deform_live=True`；`deform_live=False` 就是视图没开。

Control Source：rollout 由我们的 SDK 客户端指挥手（`--enable-hand`），会从 Pilot 手里把手
抢过来，退出时还回去（同 pi0.5 §3）。**tap 拿的是拷贝，不受 Control Source 影响。**

---

## 3 · 手上电、定手部单位、臂 bring-up、走 home、preflight、起 teleop 节点

**全部同 pi0.5 §2–§7**，命令和判据一字不差。只提醒两点：

- `collect_up.sh preflight` 里 `nothing owns :50011 -- Sharpa Pilot is not running` 这一项
  对 T-Rex **不能忽略**——Pilot 必须在跑（第 2 节）。
- 场景照样对着 `~/openarm/openarm_track/rollout/reference_start_*.png` 摆，鸡蛋要在腕相机里。

---

## 4 · Dry run —— 每次都做

```bash
# 终端 6   （已跑过 §0 的 prelude 和 ROS 三行）
cd ~/openarm/openarm_track/rollout
$V/.venv/bin/python cli.py --backend trex \
    --prompt "Pick up the egg with your left hand and place it down." \
    --trex-stats $CKPT/stats_data.json \
    --infer-lead 7 --chunk-steps 9
```

（`--infer-lead/--chunk-steps` 用第 1 节 ping 给的值。）**不带 `--enable-*` 就什么都不发。**

`--prompt` 必须**逐字**是训练 JSON 的 `language_instruction`；server 的 metadata 里有
`prompt_hint`，不一致客户端会 warning。

起来后它会先做 **2 s 触觉校准**（`calibrating tactile over 60 frames -- keep the fingers clear`）：
这 2 秒里手指不能碰任何东西。校准做的是训练转换器对每条 episode 做的同一件事：减静息偏置、
减 deform 底噪。

**判据，六条都要看**：

1. `head /dev/videoN -> 640x400`，`wrist /dev/videoM -> 640x360`（不是 256）
2. `all observation sources live` 里**五个** age（多了 `tactile`）都远小于 0.25 s
3. `tactile: ... deform_live=True, decode_errors=0`
4. 校准行：`|F| bias per finger (collector order) [~1.7 ~0 ~0 ~0 ~2.5] N, residual |F| p95 < 0.1 N, deform floor 2`
   —— 采集序是 `[pinky, ring, middle, index, thumb]`，所以 thumb 的 2–3.5 N 在**最后一个**、
   pinky 的 ~1.7 N 在第一个。residual 大于 0.3 N 说明校准时有手指碰着东西，重来。
5. `chunk 1: target[0] X.X mm from measured link7` —— 场景静止、手悬空时 **X 应是毫米级**
   （它是策略学到的"指令减实测"，也就是伺服在手的重量下的跟踪误差，中位数 8 mm）。
   几厘米以上 = 坐标系错了（`mount_geom.py` 没找到会直接拒绝启动，找到但改过要查）。
6. `infer NNN ms` 在预算内；`envelope in force: wrist step 20.0 mm / 5.73 deg, hand ...` 那行的
   手指范围来自 `stats_data.json`，不是 ±π/2。

dry run 不需要啮合：锚点用 `FK(q_meas)` 现算。

---

## 5 · 真跑

⚠️ 同 pi0.5 §9 的所有注意事项。

```bash
$V/.venv/bin/python cli.py --backend trex \
    --prompt "Pick up the egg with your left hand and place it down." \
    --trex-stats $CKPT/stats_data.json \
    --infer-lead 7 --chunk-steps 9 \
    --hand-units rad \
    --enable-hand --enable-arm
```

- `--hand-units` 用 pi0.5 §3 测出来的值。
- `--trex-stats` **每次都带**。不带会退回固件 ±π/2 钳位并打 warning。
- 校准那 2 秒手指必须悬空。
- 录制默认开着，目录 `~/openarm/rollouts/<时间戳>_<prompt>/`，`flight.npz` 里比 pi0.5 多两列：
  `wrist_pose7`（每步的绝对 link7 目标）和 `tactile_f6`（策略看到的最新一帧原始 F6）。
  视频、出图同 pi0.5 §9。

`p` 暂停、`q` 退出，🚫 不要 Ctrl-C 终端 4。

---

## 6 · 收

同 pi0.5 §10。多一条：终端 7 的 tap 最后再 Ctrl-C（它只读，先关也不会出事，但留着方便重跑）。

---

## 出问题时（只列 T-Rex 新增的）

| 现象 | 多半是 |
|---|---|
| `T-Rex needs tactile input and the observation has none` | 没走 `--backend trex`，或 tap 没起 |
| `stale observation ... {'tactile': ...}` | tap 挂了 / Pilot 关了。看终端 7 |
| `tactile sample incomplete: valid=[True, True, False, ...]` | 某个通道断流。Pilot 里重开触觉视图；还不行就 `finger_map.py` 看通道 |
| `deform_live=False` 一直不变 | Pilot 的触觉/形变视图没开 |
| 校准 residual p95 > 0.3 N | 校准时手指碰着东西，或手还在动。重跑 |
| `target[0]` 离实测几厘米 | EEF 变换错：`--mount-geom` 指错文件，或 `mount_geom.py` 的数被改过 |
| `--backend trex but the server ... advertises {...}` | 8000 口上跑的是 pi0.5 的 server |
| `the server runs with RANDOM WEIGHTS ... refusing to command hardware` | server 是 `--random_weights 1` 起的，只能 dry run |
| `chunk_steps 9 + infer_lead 7 does not fit in the backend's 16-step horizon` | 两个数加起来超过 16 |
| `inference was not ready at the chunk boundary` | 超预算。重跑 `ping_trex.py`；GPU 上有别的东西在跑 |
| 手指瞬间冲到极限 / 一动不动 | 同 pi0.5：`--hand-units` 反了 |

---

## 附：怎么选 checkpoint

Hub 上有 6 个，每 5 epoch 一个：`checkpoint-4-8215 … checkpoint-29-49290`。**没有验证集**
（模型卡说得很清楚：按样本切的 val 在 30 Hz 下泄漏相邻帧），所以离线数字不能裁决。

方法照搬 pi0.5 RUNBOOK 的附录：先用 `checkpoint-29-49290` 跑 5 次确认能动；再和
`checkpoint-19-32860` **交替**各 10 次；每次记成功/失败/失败模式/鸡蛋位置；全程录制；
只信大差距；故意把蛋放到示范里没出现过的位置看是不是过拟合。

换 checkpoint = 重起终端 1 的 server（`serve_openarm.sh <另一个目录>`），客户端
`--trex-stats` 也要换成同一目录的 `stats_data.json`——两个目录的统计量是同一个数据集算的，
理论上一样，但别赌。
