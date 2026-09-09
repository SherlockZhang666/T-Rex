# 用 OpenArm + 单只 Sharpa Wave 左手的自采数据 post-train T-Rex

日期：2026-09-09
分支：`finetune/openarm-sharpa-left`
数据：`/n/netscratch/ydu_lab/Lab/hangxing/data/tactile-steering-data/pick_up_the_egg`
起点权重：`ckpt/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`

---

## 0. 结论

**`qwen_vla/` 下的模型代码一行都不用改。** 需要写的是一个数据转换器，把 OpenArm
采集格式（`rt-v1` schema）翻译成 T-Rex post-train 的训练 JSON。跨本体的差异
（OpenArm vs Dexmate、相机 mount 位置）在 T-Rex 的动作表示里要么天然约掉，要么由
一个固定的手部安装变换吸收。

需要改的只有配置和新增文件。

---

## 1. 为什么模型不用改

### 1.1 arm 不同不影响动作空间

T-Rex 的 arm action 不是关节角，是 `utils/lerobot_common.py:66` `compute_chunk_delta_pose`
给出的 **delta-base EEF 9D**：

```
delta_xyz = R_currᵀ (t_targ − t_curr)
R_delta   = R_currᵀ R_targ
```

整条 chunk 都表达在 chunk 起始帧的 EEF 坐标系里。OpenArm 与 Dexmate 的运动学差异、
以及 base frame 约定的差异，在这个减法里被精确约掉。

**base frame 不变性的证明。** 换 base frame 相当于所有位姿左乘一个固定的 `T = (R_T, t_T)`：

```
delta_xyz' = (R_T R_c)ᵀ (R_T t_t + t_T − R_T t_c − t_T)
           = R_cᵀ R_Tᵀ R_T (t_t − t_c)
           = R_cᵀ (t_t − t_c)  =  delta_xyz
R_delta'   = (R_T R_c)ᵀ (R_T R_t) = R_cᵀ R_t = R_delta
```

`compute_tracking_error_axis_angle` 同样是 `R_stateᵀ` 形式，同样不变。

所以采集端 base frame 是 `+X=LEFT, +Y=FORWARD, +Z=DOWN`（非 REP-103，见采集方
`SCHEMA.md:95`）这件事，**对 action 和 tracking_error 完全没有影响，不会产生镜像
倒置的轨迹**。它唯一影响的是 `state` 里那个绝对 9D 位姿；而 `state` 在
`--use_robot_state 0` 时不进网络，且跨本体本来就不可迁移（T-Rex 的 base 是 Dexmate
自己的躯干系）。转换器仍会把 state 转成 REP-103 存档，纯粹为了去掉一个雷。

### 1.2 EEF 坐标系必须对齐（这个会影响 `R_delta`）

base frame 约掉了，但 **EEF 系不会** —— 它以共轭的形式出现在 `R_delta` 里。所以必须
把采集端的位姿外推到 T-Rex 所用的那个 EEF 系。

**T-Rex 的 `L_ee` 就是 Sharpa 手的 `hand_wrist` link。** 三行代码为证：

| 证据 | 位置 |
| --- | --- |
| 加载的是 `left_sharpa_wave_with_wrist.xml`，不是 `with_flange` | `hardware_code/teleop/robot_descriptions.py:66-69` |
| 该模型根 link 名为 `left_hand_wrist` | `third_party/sharpa-urdf-usd-xml/wave_01/left_sharpa_wave/left_sharpa_wave_with_wrist.urdf:6` |
| 手以 **单位 SE3** 挂在 `L_ee` 上：`pin.appendModel(..., left_ee_frame_id, SE3(I, [0,0,0]))` | `robot_descriptions.py:305-310` |

采集端的 `obs/wrist_pose_b` / `action/wrist_pose_b` 记的是 `openarm_{side}_link7`
（采集方已用 FK 逐帧核对，误差 0.0000 mm；归档 attr `urdf =
openarm_v10_shoulder_bimanual.urdf` 的左链到 link7 就结束，手只作为点质量 payload
参与 `g(q)`，没有任何路径能让手部几何漏进这两个字段）。

于是完整链条为：

```
T_link7→L_ee = Rz(PALM_ROLL_DEG) · Trans(0, 0, HAND_MOUNT_Z + FLANGE_TO_WRIST_Z)
             = Rz(+144.175°)     · Trans(0, 0, 0.0540 + 0.0005 = 0.0545)      [左臂]
```

`FLANGE_TO_WRIST_Z = 0.0005` 这 0.5 mm **必须串上**：采集端复合模型挂的是
`with_flange`（根 link `hand_flange`），而 `with_flange.urdf:35-41` 的
`left_hand_flange_fix_joint` 正是 `hand_flange --(0,0,0.0005)--> hand_wrist`。T-Rex
用的是后者。

`Rz` 与 `Trans(0,0,z)` 对易，写哪个顺序都一样。

数值的唯一真源是采集仓库的
`openarm-sharpa-sim/openarm_sharpa/assets/mount_geom.py`（本机已有）。
`HAND_MOUNT_Z = 0.0540` 是 2026-08-24 切掉 link7 夹爪电机之后的值；本批数据采于
09-02/09-03，在切之后，用 0.0540 正确。08-24 之前的归档不能用这套几何。

**这个变换只作用于左臂。** 右臂没有手（`embodiment = openarm_bimanual_v10_sharpa_left`），
engaged 帧数为 0，在下面 2.1 的方案里右臂那 31 维根本不来自采集数据，转换器不对它
做任何 FK。

### 1.3 手是同一只

同型号 Sharpa Wave 22-DoF。`robot_descriptions.py:78+` 的 `SHARPA_HAND_JOINT_ORDER`
以 `thumb_CMC_FE` 开头，与采集端 `attrs["hand_joint_names"]` 的 22 个名字逐一对应。
hand action 那 22 维直接搬。

### 1.4 相机 mount 位置不同不需要改代码

ViT 的输入就是一张图。视角差异由 finetune 吸收，没有任何一处代码对相机外参有假设。
唯一需要设的是 `--image_size`（现为 `384 288`）和一个可选的 head crop。

---

## 2. 需要做的事

### 2.1 保持 `action_dim=62`，右半边写常数 padding

**不要把 `--action_dim` 改成 31。** `scripts/train.py:834-841` 在 resume 时是
**静默丢弃 shape 不匹配的 key**：

```python
for k, v in resume_sd.items():
    if k in model_sd and model_sd[k].shape != v.shape:
        skipped.append(k)          # ← 只打印一行，然后继续
```

改成 31 会让 `x_embedder`、`final_layer`、`final_layer_tactile`、`state_embedder`
全部落进 `skipped`，随后被 `initialize_vla_weights()` 重新随机初始化 —— midtrain 学
到的 action head 就扔掉了，而训练日志看起来完全正常。

所以：

| 维度区间 | 内容 |
| --- | --- |
| `action[0:9]` / `state[0:9]` | 左臂 EEF 9D（真实） |
| `action[9:31]` / `state[9:31]` | 左手 22 关节（真实） |
| `action[31:40]` / `state[31:40]` | **单位位姿** `[0,0,0, 1,0,0, 0,1,0]` |
| `action[40:62]` / `state[40:62]` | 22 个 0 |

右半边写单位位姿而不是全零，是因为全零不是合法旋转，会让任何 replay / 可视化工具
画出垃圾。数值上两者都被 `_normalize` 映射成常数 −1（`q99 − q01 = 0`，分母 `+1e-8`），
等价。

**触觉同理必须保持 10 指 × 6。** `qwen_vla/modeling_vla.py:137-145` 的
`tacf6_vqvae_min/max/mask` 是**写死 60 维的 buffer**，`encode_tactile_f6_history`
里 `n_hands = NF // 5`。喂 5 指会直接崩。所以 `f6[0:5]` 真实、`f6[5:10] = 0`；
deform 前 5 路真实、后 5 路全零帧。

零是右半边的正确填充值：去偏置后"无接触"就是 0，去底噪后 T-Rex 的 deform 背景也
恰好是 0。恒定输入经 VQ-VAE / DeformEncoder 得到一个恒定 token，无害。

### 2.2 数据格式：走 JSON，不走 LeRobot

两条路都被 `scripts/train.py` 支持（`--data_format json|lerobot`）。选 JSON：

1. **依赖**。`Downloads/envs/trex` 里没有 `lerobot`、没有 `av`；JSON 路径只需要
   h5py + cv2 + numpy，全都有。
2. **这是作者 post-train 实际用的路径**，`scripts/train.sh` 默认就是它。
3. **单腕相机天然可解**。`input_image_fast` 是一个路径列表，把同一个左腕图的路径
   写两遍即可复现 midtrain 的双图布局，不需要真的复制一份视频。

顺带记录一个事实，免得以后再踩：**HF 上开源的 midtrain 数据集不是训练器能读的格式。**
`data/trex_dataset/meta/info.json` 里 `action`/`state` 是 **58 维绝对关节角**
（7 arm joint + 22 hand，双臂），key 叫 `observation.images.head_left`、
`observation.tactile_force`。而 `qwen_vla/lerobot_dataset.py` 要的是
`observation.images.head`、`observation.tactile_f6 [10,6]`、已烘焙的
`action = [16,62]` delta-base chunk、10 路 `observation.tactile_deform.*`。训练器读
的是 `utils/convert_inlab_to_lerobot.py` 的输出，不是 HF 上那份。post-train 数据集
未开源。

### 2.3 JSON 路径的硬性格式约束

`scripts/train.py` 里有两处**写死的正则**，转换器的输出目录结构必须精确满足，否则
会静默退化：

| 正则 | 位置 | 后果 |
| --- | --- | --- |
| `(.+/episode_\d+)/image(\d+)_` 作用于 `tactile_image_deform[0]` | `train.py:177`、`train.py:209` | 匹配不上 → VQ-VAE 的 F6 历史窗退化成"把当前帧重复 16 次"，触觉时序信息全丢，而且**不报错** |
| `re.sub(r'image\d+_', f'image{idx}_', slow_path)` | `train.py:299` | 匹配不上 → FLARE 未来帧全部取不到，`_load_flare_frame` 返回 `None` |

因此文件命名必须是：

```
<img_root>/<task>/episode_<NNNN>/image<frame>_head.png
<img_root>/<task>/episode_<NNNN>/image<frame>_wrist_left.png
<img_root>/<task>/episode_<NNNN>/image<frame>_tactile_left_deform_<0..4>.png
<img_root>/<task>/episode_<NNNN>/image<frame>_tactile_right_deform_<0..4>.png
```

`<frame>` 必须是 episode 内连续递增的整数（FLARE 靠 `idx + (k+1)*stride` 直接算
文件名，中间缺号就取不到帧）。

统计量文件路径由 `train.py:126` 写死为 `data_path.replace(".json", "_statistics.json")`。

deform PNG 存 0-255 uint8，`_open_gray`（`train.py:275`）会除以 255。

### 2.4 转换器里逐条处理的坑

每一条都会静默产生错误结果，没有一条会抛异常。

| # | 问题 | 依据 | 处理 |
| --- | --- | --- | --- |
| 1 | **手指顺序是反的** | 采集端 `tactile_finger_order = [pinky, ring, middle, index, thumb]`（实测标定，见采集文档 §3）；T-Rex 是 `[thumb, index, middle, ring, pinky]`（开源数据集 key 名 + `gen_json` 写图顺序） | f6 和 deform 都按索引反转 |
| 2 | **thumb 通道有 2.2–3.5 N 零偏** | 采集文档 §8.1 | 按 episode 减 `attrs["tactile_zero_offset"]["magnitude_N"]`。**这条不是可选的**：embedded VQ-VAE 用的是 checkpoint 里烘死的 `tacf6_vqvae_min/max` buffer（`modeling_vla.py:137-145`），不是我们重算的 q01/q99，不减偏置 thumb 会落到码本的另一个区域 |
| 3 | **deform 有常数底噪 2**，T-Rex 背景恰好是 0 | `tools/check_our_deform.py` 实测 | 取众数底减掉（沿用 `check_our_deform.py` 的做法，不写死 2） |
| 4 | 哪个是标签 | 采集文档 §4 | `action/wrist_pose_b`（quat **wxyz** → 4×4）当 `*_arm_target_pose`；`obs/wrist_pose_b` 当 `*_arm_current_pose`；`action/hand_joint_pos` 当 hand 标签（**不是** `hand/joint_pos`，后者是读数，抓紧后会饱和） |
| 5 | 腕相机不在行时钟上（60 fps free-run） | 采集文档 §6 | 按 `cam_wrist/frame_index[i]` 取帧；`-1`（该行之前还没有腕帧到达）用第一个有效帧 hold |
| 6 | 右臂列全是 NaN（by design） | 采集文档 §4 | 从不读取；被 2.1 的常数 padding 覆盖 |
| 7 | 无效行 | `obs/valid`、`hand/valid`、`tactile/deform_valid` | 按最近有效值 hold |
| 8 | 该丢的 episode | `index.jsonl` 的 `discarded`、`teleop/engaged[left]` | 过滤（本批 38 条全部 `discarded: false` / `success: true`） |
| 9 | head crop | — | 默认不 crop（T-Rex 的 `CROP_BOX_SLOW` 是针对 Dexmate 头相机视野标的，对我们的 mount 没有意义）。留 `--crop_box` 参数 |

### 2.5 EEF 外推的调用约定

不做成"默认 identity 的 `--ee_offset`"。默认 identity 意味着某次忘了传参就会静默
产出 link7 系的数据，而这个错误在下游看起来完全正常 —— 只是所有轨迹整体偏 54.5 mm
外加一个 144.175° 的滚转。

改成：`--mount_geom <path>` 按文件路径 import `mount_geom` 模块并算出变换，默认指向
同级的 `openarm-sharpa-sim`；**找不到就硬报错**。逃生口是显式的 `--no_ee_offset`。
解析出的数值打印到日志并写进输出 JSON 的同级 `_provenance.json` 存档。

---

## 3. 新增文件

| 文件 | 作用 |
| --- | --- |
| `utils/gen_json_openarm_sharpa_left.py` | 转换器主体。session 目录 → 训练 JSON + `_statistics.json` + 图片树 |
| `utils/gen_json_openarm_sharpa_left.sbatch` | 转换作业（CPU） |
| `tools/verify_openarm_json.py` | 转换后的断言检查 |
| `tools/verify_openarm_json.sbatch` | 验证作业 |
| `scripts/train_openarm.sh` | 训练启动脚本（从 `train.sh` 派生，改掉残留的 NVIDIA 集群路径） |

`qwen_vla/` 不动。`scripts/train.py` 不动。

所有 SLURM 作业指定 `#SBATCH -A hankyang_lab`，不在 login 节点跑。

---

## 4. 验证（上 GPU 之前）

`tools/verify_openarm_json.py` 逐条断言：

1. **往返**：从 delta chunk + base pose 重建绝对目标位姿，与源 `action/wrist_pose_b`
   比 → 应为 float32 噪声级
2. **偏置生效**：相对 link7 的位移中位数 = 54.5 mm，绕 Z 转角 = 144.175°
3. **手指序**：断言重排后 index 0 映射到 thumb（采集端 ch9）
4. **去偏置后** thumb 在静止段的中位数 ≈ 0
5. **deform 众数像素 == 0**
6. **norm stats**：右半 `q01 == q99`；左半不退化
7. **正则可匹配**：用 `train.py` 里那两个正则实际跑一遍输出的路径
8. **FLARE 连续性**：随机抽样本，确认 `image{idx + k*stride}_head.png` 都存在
9. 再跑一遍已有的 `tools/check_our_deform.py`，确认转换后的 deform 仍落在 T-Rex 分布内

然后 1 卡 `--n_epochs 1` 冒烟。**唯一必须盯的一行是 resume 时打印的
`Skipped 0 keys with shape mismatch`** —— 只要非 0，2.1 的 padding 方案就是错的。

---

## 5. 训练配置

以 `scripts/train.sh` 为基准，改动：

| 参数 | 值 | 理由 |
| --- | --- | --- |
| `--action_dim` | `62`（不变） | 见 2.1 |
| `--use_robot_state` | `0`（不变） | 绝对 state 跨本体不可迁移 |
| `--resume_source` | `midtrain` | 保住 tactile expert |
| `--data_format` | `json` | 见 2.2 |
| `--n_epochs` / `--learning_rate` | 待定 | 38 条 episode / 15792 帧 / ~8.8 分钟，比作者的 post-train 量小得多，现在的 `100` / `1e-4` 大概要往下调，等冒烟跑出 loss 曲线再定 |

`--action_chunk 16`、`fps 30` 与采集端行时钟（30 Hz）天然对上。

**代码写完即停，训练由人手动起。**

---

## 6. 数据清单

`tactile-steering-data/pick_up_the_egg`，3 个 session：

```
20260902_214037_pick_up_the_egg
20260903_000158_pick_up_the_egg
20260903_002436_pick_up_the_egg
```

38 episodes / 15792 frames / ~8.8 分钟，全部 `success: true`、`discarded: false`。
采集日期在 2026-08-24 的 link7 改动之后，`HAND_MOUNT_Z = 0.0540` 适用。
