# 在 OpenArm + 单只 Sharpa Wave 左手真机上 rollout 训好的 T-Rex

日期：2026-09-11
分支：T-Rex `rollout/openarm-sharpa-left`（自 `finetune/openarm-sharpa-left` 派生）；
rig `~/openarm/openarm_track` 分支 `trex-backend`（自 `policy-runtime` 派生）
checkpoint：`zhx-tactile-steering/T-Rex-openarm-sharpa-left-egg`（6 个，每 5 epoch 一个）
前身：pi0.5 的同一套 rollout，见 `~/Public/openpi/RUNBOOK.md` 与 `~/openarm/openarm_track/rollout/`

---

## 0. 结论

**rig 的 rollout 客户端（`openarm_track/rollout/`）保留，T-Rex 作为它的第二个 policy backend 接进去。**
所有硬件层（相机、手、臂 UDP 桥、三层安全包络、录制、30 Hz 调度）原样复用；新写的只有
四件事：

1. **T-Rex 侧一个 websocket 推理服务**（`hardware_code/openarm/serve.py`），把 `scripts/test.py`
   已有的 `CascadedServer` 用 openpi 同款的 websocket + msgpack 传输暴露出来，这样 rig 侧现成的
   `openpi_client.WebsocketClientPolicy`、`ping` 工具、6 终端流程全部沿用。
2. **rig 侧一个 backend**（`rollout/policy_trex.py`）：拼观测 payload，把 62 维 delta-base 动作
   解码成 link7 系的**绝对**腕部目标 + 22 维手指目标。
3. **rig 的 seam 扩一个口**：`RobotCommand` 允许携带绝对腕部目标；`robot_openarm.apply` 把它换算成
   相对当前指令位姿的增量后走**原有的**单步/累积包络。pi0.5 路径一个字节不变。
4. **触觉观测**：`observe.Sources` 可选挂一个 `SniffTactile`（采集时就在用的、只读的 tap 客户端），
   每步把最近 16 帧 F6 与当前 deform 图塞进 `Observation.tactile`。

模型代码 `qwen_vla/` 与 `scripts/test.py` 不改。

---

## 1. 为什么是这个架构

### 1.1 沿用 rig 客户端而不是移植 `hardware_code/eval/eval_trex_async.py`

`eval_trex_async.py` 是给 Dexmate 双臂 + dexcontrol 写的：IK 是 pink、手是 dexcontrol 的 API、
相机是 ZMQ 推流。我们的臂由 `openarm_vr_teleop.py` 走 UDP 收笛卡尔目标，手走 Sharpa SDK，
这些 rig 已经封好了，且带着 pi0.5 那轮踩出来的所有护栏（`--home-on-start`、engage-gap、
单步/累积包络、stale-observation 门禁）。移植 eval 脚本等于把这些重写一遍。

`eval_trex_async.py` 里值得搬的只有两条语义，都在 §3 里落实：动作相对 chunk 起始时刻的
**实测** EEF 位姿解码；每个 chunk 开头发 `slow_and_fast`。

### 1.2 传输层用 openpi 协议而不是 `test.py` 的 ZMQ + pickle

`test.py` 的 ZMQ 服务是 pickle 传 numpy。server 跑 Python 3.10 / numpy 2.2，rig venv 是
Python 3.12 / numpy 2.5，pickle 跨版本能不能解是运气。而 rig 已经为 pi0.5 装好
`openpi_client`（websocket + msgpack-numpy），`ping_policy.py`、`cli.py` 的连接逻辑都基于它。
T-Rex 侧 vendor 40 行 `msgpack_numpy.py`（Apache-2.0，来自 openpi-client）即可对上。

`CascadedServer` 本身（模型加载、resize、slow/fast 级联、反归一化、F6 滚动窗）原样复用，
`serve.py` 只做 "msgpack payload → `predict()` 入参" 的翻译。

### 1.3 一个 chunk 一次推理，不做 chunk 内 fast refine

`eval_trex_async.py` 在 chunk 内的 4/8/12 步再发 `fast` 用新触觉精修。rig 的
`PolicyBackend.infer` 是"一次调用一整个 chunk"，loop 每 `--chunk-steps` 步重新推理。
第一版不加 chunk 内 refine：

- 我们每 8 步就重推一次（见 §5），等价于每 267 ms 用新触觉更新一次，和 refine 的 4 步间隔
  只差一倍；
- refine 需要 loop 支持"chunk 中途替换剩余动作"，属于 loop 的改动，先不碰；
- 先证明主链路能动，再评估 refine 值不值。

---

## 2. 观测契约（client → server）

msgpack dict，键固定：

| 键 | 类型 | 含义 |
|---|---|---|
| `prompt` | str | **逐字** `Pick up the egg with your left hand and place it down.`（`gen_json_openarm_sharpa_left.sbatch`、模型卡）|
| `head` | `(400, 640, 3)` uint8 RGB | 场景相机。训练时 1280×800 → `INTER_AREA` 到 640×400 存 PNG，再由 `test.py` LANCZOS 到 384×288。rig 相机改用 `short_side=400` 直接出 640×400，不再是 pi0.5 的 256 |
| `wrist` | `(360, 640, 3)` uint8 RGB | 左腕相机。训练 960×540 → 640×360。server 把它同时填进两个 fast 槽（训练就是这么重复的）|
| `tactile_f6` | `(16, 5, 6)` float32 | 最近 16 帧（30 Hz 行时钟，零阶保持），**T-Rex 手指序 [thumb, index, middle, ring, pinky]，已减零偏**。server 右手补零到 `(16, 10, 6)` |
| `tactile_deform` | `(5, 240, 240)` uint8 | 当前帧，T-Rex 手指序，**已减底噪**（背景为 0）。server 右手补 5 张全零 |
| `state` | `(62,)` float32，可选 | 该 checkpoint `use_robot_state=0`，server 忽略；留着是为了 `record` 对得上训练 JSON |

反方向：

| 键 | 类型 |
|---|---|
| `actions` | `(16, 62)` float32，已按 `stats_data.json` 的 q01/q99 反归一化 |
| `chunk_id` | int |
| `server_timing` | `{"infer_ms": float}`，与 openpi 相同 |

server 启动时发的 metadata 带 `{"model": "trex", "horizon": 16, "action_dim": 62,
"prompt_hint": ..., "checkpoint": ...}`，client 用 `horizon` 校验。

**数据集约定（手指序、零偏、底噪）的代码放在 T-Rex**：`hardware_code/openarm/tactile_convert.py`，
从 `utils/gen_json_openarm_sharpa_left.py` import `COLLECTOR_TO_TREX`、`deform_floor`，
保证和转换器用的是同一份常量。rig backend 通过 `--trex-root` 把 T-Rex 根目录加进
`sys.path` 来 import 它。

---

## 3. 动作契约与解码

### 3.1 62 维布局

`[0:9]` 左臂 delta-base EEF 9D，`[9:31]` 左手 22 关节绝对弧度（`thumb_CMC_FE` 起），
`[31:62]` 右侧常数 padding，**丢弃**。

### 3.2 从 delta 到绝对目标

训练标签（`gen_json_openarm_sharpa_left.py:465-475`，`frame_stride=1`）：

```
action[k] = compute_chunk_delta_pose(T_meas(r), T_cmd(r + k))       k = 0..15
delta_xyz = R_measᵀ (t_cmd − t_meas)      R_delta = R_measᵀ R_cmd
```

其中 `T_meas(r)` = 观测时刻 `obs/wrist_pose_b` 的 FK（**实测**关节角），`T_cmd` =
`action/wrist_pose_b`（teleop 节点的笛卡尔**指令**），两者都先右乘 `T_mount` 换到
`hand_wrist` 系。所以解码是：

```
T_ee_meas  = FK_link7(q_meas at obs) · T_mount            T_mount = Rz(+144.175°)·Trans(0,0,0.0545)
T_ee_k     = T_ee_meas · [R_delta_k | delta_xyz_k]         R_delta_k 由 6D 两列 Gram-Schmidt 重建
T_link7_k  = T_ee_k · T_mount⁻¹
```

`T_mount` 的数值 **从 `~/sim/openarm-sharpa-sim/openarm_sharpa/assets/mount_geom.py` import**，
不写死（和转换器同一真源）；找不到就报错退出，逃生口 `--mount-geom PATH`。

base frame 在这条链里约掉（设计文档 §1.1 的证明），所以直接用 rig 的 pinocchio base，
它就是节点 `apply_base_delta` 的那个系。

### 3.3 绝对目标怎么进 rig 的 PalmChain

rig 的 `RobotCommand` 是"相对**上一个指令**位姿的增量"，pi0.5 的训练标签就是这么定义的。
T-Rex 的标签是"相对**观测时刻实测**位姿的绝对目标"。两者不能硬换：把 T-Rex 目标当增量
会把重力下沉量每个 chunk 多算一次。

正确做法：`RobotCommand` 增加可选字段 `wrist_pose7`（link7 在 base 系的绝对目标，w-first 四元数）。
`robot_openarm.apply` 看到它时，用 chain 当前的指令位姿现算增量

```
dp = R_chainᵀ (p_target − p_chain)        dr = log(R_chainᵀ R_target)
```

然后**走原来的路**：`_check_step(dp, dr)` → 手 → `chain.step(dp, dr)` → 累积包络 → publish。
chain 落到的位置就是目标本身（差一个正交化）。pi0.5 的 `wrist_pose7=None` 路径完全不变。

这样做还顺手解决了 `--infer-lead` 的语义：`loop.py` 从索引 `lead` 开始消费新 chunk，因为
`action[j]` 是"观测后第 j 步的目标"——对绝对目标这正是对的，且不存在增量被重复积分的问题。

指令位姿 vs 实测位姿的一致性：训练里目标 = 实测 ⊕ 学到的偏移（含伺服跟踪误差，中位数 8 mm），
节点收到目标后臂又沉到实测。稳态下 `target_k ≈ chain.pose`，增量小；engage 后第一步的增量
= (节点 hold 目标) 到 (实测 ⊕ 学到偏移) 的差，在 20 mm 单步护栏之内。

### 3.4 手指

`actions[:, 9:31]` 直接给 `RobotCommand.hand_joints`；关节序在训练时已验证与
`episode_writer.HAND_JOINT_NAMES` 逐一相同（设计文档 §4b，`max |diff| = 0`）。

### 3.5 `suggested_limits`

从 checkpoint 的 `stats_data.json` 取 `action.q01/q99`（形状 `[16, 62]`），手指包络 =
16 步上的 `min(q01[:, 9:31]) − margin` / `max(q99[:, 9:31]) + margin`。腕部单步包络**不从
统计量推**：T-Rex 的统计量是相对 chunk 起点的，不是相邻两步之差，硬推会得到一个过宽或
过窄都说不清的数。腕部沿用 `DEFAULT_HARD_LIMITS` 的 20 mm / 5.7°，训练集相邻目标之差的上限
（4.6 mm / 1.3°）仍在它的 1/4 处。

---

## 4. 触觉流水线

```
sharpa_tap.py --serve /tmp/sharpa_tap.sock   (sudo, 采集时同一个)
      │  AF_PACKET 抄一份 :50011，不抢 Pilot 的口
      ▼
SniffTactile(with_state=False)                (vr/collect/tactile.py，采集时同一个类)
      │  .tactile.value = {"f6": (5,6) 采集序, "deform": (5,240,240) u8, "valid", "deform_valid"}
      ▼
observe.Sources.read()  @30 Hz
      │  deque(maxlen=16) 存原始 f6 → Observation.tactile = {"f6_window": (16,5,6), "deform": ..., ...}
      ▼
policy_trex.TrexBackend.infer()
      │  tactile_convert: 采集序→T-Rex序，f6 − bias(5,6)，deform − floor
      ▼
serve.py：右手补零 → CascadedServer.predict("slow_and_fast")
```

- **手指序**：采集端实测 `[pinky, ring, middle, index, thumb]`，T-Rex `[thumb, …, pinky]`，
  `COLLECTOR_TO_TREX` 反转。f6 与 deform 都反。
- **零偏**：训练按 episode 取首次接触前各帧的**均值（全部 6 通道）**减掉（`f6_zero_offset`）。
  rollout 在 loop 开始前、手悬空未接触时采 2 s（60 帧）取均值作为本次 run 的 bias。
  这一步**必须**做：VQ-VAE 的 min/max 烘死在 checkpoint 里，thumb 2.2–3.5 N 的偏置不减会落到
  码本另一个区域。
- **底噪**：训练按 episode 取众数像素减掉（实测 2）。rollout 从同一段校准帧取众数。
- **窗口密度**：VQ-VAE 的 16 帧窗口是 30 Hz 相邻行。client 自己维护 30 Hz 的 deque 并把整窗
  送过去；不能让 server 用"每次推理推一帧"的滚动缓冲——那样帧距是 8 步而不是 1 步。
- **右手**：F6 全零、deform 全零，与训练 padding 一致。
- Pilot 需要开着且 tactile/deform 视图打开（设备只在有人请求时才算 deform），和采集时一样；
  pi0.5 那份 RUNBOOK 里"Pilot 不用开"对 T-Rex **不成立**。

---

## 5. 调度与延迟预算

horizon = 16（`action_chunk`），loop 的约束 `chunk_steps + infer_lead ≤ 16` 且 `infer_lead < chunk_steps`
→ 最大 `--infer-lead 7 --chunk-steps 9`，推理预算 **233 ms**（含 slow + fast 两段与网络）。

4.25 B 的 MoT、10 步 Euler、两张 384×288 图，在 5090 Laptop 上多少毫秒**没有测过**。
第一步就是测（`ping_trex.py`，可以用随机权重的模型建图测时延，不用等 8.5 GB 下完）。

若超预算，备选按顺序：

1. `--cascaded_total_steps` 降到 6–8（split 相应缩），牺牲少量精度；
2. **动作步长 2**：loop 以 15 Hz 执行 `action[0,2,4,…]`。对绝对目标这是精确的（第 2k 个动作就是
   2k/30 s 后的目标），预算翻倍到 467 ms。需要给 `loop.py` 加 `--action-stride`，第一版**不做**，
   只在测出来确实需要时再加。

---

## 6. 安全

三层包络（单步腕、累积腕、手指钳位/限速）和 stale-observation 门禁**全部不动**，绝对目标在
`apply` 里换算成增量后走同一条检查。新增的风险点只有两个，都在 dry run 里就能看见：

- **T_mount 或 6D→R 的手性错**：表现为腕部目标离实测 5 cm 以上或第一步就撞单步护栏。
  dry run 的每秒日志打 `target − FK(q_meas)` 的距离，静止场景下应是毫米级。
- **零偏没减**：表现为手指策略"以为一直在碰东西"。dry run 打印校准后的 thumb 静息 |F|，应 < 0.1 N。

---

## 7. 文件清单

### T-Rex（本分支）

| 文件 | 作用 |
|---|---|
| `hardware_code/openarm/serve.py` | websocket + msgpack 推理服务，包 `scripts/test.py` 的 `CascadedServer` |
| `hardware_code/openarm/msgpack_numpy.py` | openpi-client 的 40 行序列化，vendor |
| `hardware_code/openarm/tactile_convert.py` | 采集约定 → 数据集约定（手指序 / 零偏 / 底噪 / 右手补零），纯 numpy |
| `hardware_code/openarm/ping_trex.py` | 合成观测打 server：时延、chunk 形状、有限性、`--random-weights` 免 checkpoint 测时延 |
| `hardware_code/openarm/RUNBOOK.md` | 操作单（中文），与 pi0.5 那份同结构，只写差异处 |
| `hardware_code/openarm/README.md` | 契约与坐标约定（英文，随代码） |
| `scripts/serve_openarm.sh` | 从 checkpoint 目录起 server，参数从 `training_args.json` 推，缺 `model.pt` / `stats_data.json` 直接拒绝 |
| `scripts/setup_rollout_env.sh` | 5090 Laptop（sm_120）的推理 venv：torch 2.7.1+cu128 |

### rig（`~/openarm/openarm_track`，分支 `trex-backend`）

| 文件 | 改动 |
|---|---|
| `rollout/contracts.py` | `Observation.tactile: dict \| None`；`RobotCommand.wrist_pose7: np.ndarray \| None`；`Applied.wrist_dp/dr` |
| `rollout/robot_openarm.py` | `apply` 支持绝对目标（§3.3） |
| `rollout/observe.py` | `Sources(tactile=...)`：age 检查、16 帧 deque、`Observation.tactile` |
| `rollout/policy_trex.py` | backend（§2、§3） |
| `rollout/record.py` | 记 `wrist_pose7` 目标、实际积分的增量、F6 (5,6) |
| `rollout/cli.py` | `--backend trex`、`--trex-root`、`--tactile-sock`、`--trex-stats`、`--mount-geom`、per-camera 尺寸、零偏校准 |
| `rollout/*_test.py` | 新增 `policy_trex_test.py`，扩 `robot_openarm_test.py` / `contracts_test.py` / `observe_test.py` |

---

## 8. 验证顺序

1. rig 单测（openpi venv 的 pytest，和现有测试一样）：绝对目标 → 增量 → chain 落点 = 目标；
   pi0.5 路径回归不变；解码往返（`compute_chunk_delta_pose` 的逆）；触觉转换与 `gen_json` 一致。
2. `ping_trex.py --random-weights`：不等 checkpoint，先拿时延，定 `--infer-lead/--chunk-steps`。
3. checkpoint 到位后 `ping_trex.py`：chunk `(16, 62)`、有限、手指在 q01/q99 内。
4. dry run（不带 `--enable-*`）：看 §6 的两条。
5. 真跑：先 `checkpoint-29-49290` 跑 5 次，再和 `checkpoint-19-32860` 交替 A/B——方法照搬
   pi0.5 RUNBOOK 附录。

---

## 5b. 时延实测（2026-09-11，`checkpoint-19-32860`，RTX 5090 Laptop，平台功耗上限 ~95 W）

| server | 往返 p50 / p95 | slow / fast | 放得下 233 ms？ |
|---|---|---|---|
| 训练参考 `10/6` | 320 / 333 ms | 240 / 72 | 否，任何 lead 都不行 |
| `--cascaded_total_steps 5 --cascaded_split_step 3` | 213 / 224 ms | 164 / 43 | 是，余 9 ms |

每步 Euler 约 27 ms（两张腕图槽 + chunk ≈ 230 token 过 28 层 MoT，batch 1，launch-bound），
prefill ~40 ms，ViT ~24 ms。§5 里"动作步长 2"的备选是错的：上限是 horizon 的一半（wall time），
步长不改变它。可行的两条：`5/3` @ 30 Hz（τ_split 仍是训练的 0.4，触觉专家只在 τ ≤ 0.4 训过，
比例不能变）；或 `10/6` @ 15 Hz 慢动作。真机上两种都要试。

## 9. 已知未决

- **下载**：HF CDN 到本机实测 ~150 KB/s，两个 8.5 GB 要十几个小时。若集群侧有副本，
  rsync 更快。
- **时延**未测，§5 的预算是否成立取决于它。
- chunk 内 fast refine 不做（§1.3）。
- `state` 不进网络，但 `hardware_code/openarm/README.md` 写清了 62 维绝对位姿的 REP-103 约定，
  以后开 `use_robot_state` 时 client 已经能填。
