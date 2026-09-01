# Tactile Deformation Encoder — 训练、验证与跨项目复用

论文只给了架构（ResNet-18 改单通道输入、保留前三个 residual stage、每级接 3×3 卷积
投到 128 通道）和一句 "pre-trained within a self-supervised convolutional autoencoder
framework and subsequently frozen"。**没有损失函数、超参、数据量，也没有任何评估指标。**
仓库里同样没有 —— `qwen_vla/DeformAE.py` 只有 `DeformEncoder`、一个 `deformation_head`
和一个用 `cv2.imshow` 肉眼看重建图的 `__main__` demo，decoder 的训练代码未发布。

本文档记录 `tools/` 下四个脚本各自解决什么问题、判据从哪来、以及在自采数据上的实测结论。

---

## 1. 两条获取权重的路径

| | 自己训 | 从官方 checkpoint 抠 |
|---|---|---|
| 脚本 | `extract_deform_frames.py` → `train_deform_ae.py` | `extract_deform_encoder_from_ckpt.py` |
| 产物 | `my_sharpa_wave_deform_encoder.pth`（fp32，含 decoder，13.7 MB） | `sharpa_wave_deform_encoder.pth`（bf16，仅 encoder，6.4 MB） |
| 训练量 | `num_batches_tracked` = 10,893 | `num_batches_tracked` = 4,226,018 |

**默认用官方那份。** 它和 `T-Rex_midtrain_*` 里的 `deform_proj` 及下游 tactile expert 是
共同训练出来的，特征空间对齐；换成自训的那份，encoder 输出分布完全不同，而 `deform_proj`
仍是 midtrain 学到的那个。自训的那份留作对照实验（例如验证 encoder 质量对下游成功率的影响）。

两份权重的平均 cosine 相似度只有 **0.25** —— 自编码器解空间有巨大的置换/旋转对称性，
两次独立训练本就不会收敛到同一组权重。**所以不能靠"和作者的像不像"判断自训版对不对。**

### 1.1 关于 bf16

发布的 `model.pt` 是 8.5 GB，全部 1412 个顶层键里 1360 个是 bf16（4.251 B 参数 × 2 字节），
33 个 fp32 的是归一化 buffer，**没有 optimizer state、没有 fp32 master weights**。
所以 fp32 精度在作者存盘时就已丢失，抠不出来。

但这件事影响接近于零。bf16 → fp32 的转换本身是**无损的**（bf16 就是 fp32 截掉后 16 位尾数）。
把自训的 fp32 权重走一遍 `fp32 → bf16 → fp32` 精确模拟这个损失，在真实形变帧上实测：

| | |
|---|---|
| 权重相对误差 | 中位数 1.4e-3，p99 3.5e-3 |
| 特征相对 L2 差异 | 5.1e-3 |
| 特征 cosine 相似度 | **0.99998879** |

比训练随机性带来的差异（cosine 0.25）小几万倍。冻结推理和 finetune 都不受影响。

---

## 2. 自监督训练配方（`train_deform_ae.py`）

Job 39447153，A100 单卡 ~65 min，30 epoch。

- **数据**：`extract_deform_frames.py` 顺序解码，每 500 帧留 1 帧。94 个分片 →
  train 99,830 帧 / 82 文件，val 9,687 帧 / 12 文件。
- **验证集按整文件划分，不是随机帧** —— 30 fps 相邻帧近似重复，随机划分会泄漏。
- **输入尺度 `[0,1]`**（除以 255）。两条 VLA 数据路径都是这个尺度
  （`scripts/train.py:277` 除以 255；`lerobot_dataset.py:217` 取 [0,1] 视频张量的通道 0），
  `modeling_vla.py:508` 不再缩放。`DeformAE.py` 的 `__main__` demo 喂 0-255，**是错的**。
- **超参**：`lr=3e-4`、warmup 500 步 + cosine、grad clip 1.0、bf16 autocast、batch 256、
  AdamW wd 1e-4。
- **`--decoder_act leaky`**：原版 decoder 的 ReLU 在第一个 epoch 全死（`lr=1e-3`，job 39302558），
  输出恒定图像、encoder 收不到梯度、loss 卡死 29 轮。decoder 训完即弃，换激活不影响 encoder 兼容性。
- **`--mag_alpha 10`**：接触像素只占约 1%，朴素 MSE 会奖励"只预测背景"。
  权重 `w = 1 + 10·relu(x − 0.03)`。

### 2.1 判据（全部自定，论文未提供）

| 判据 | 为什么 | 实测 |
|---|---|---|
| `out_std` ≥ 1e-4，连续两轮不达标即中止 | **最重要**。常数图像的 PSNR 也有 24.3 dB，光看 PSNR 会被骗 | 全程 0.018，未塌缩 |
| R² vs 数据方差 | 常数预测器 R² ≤ 0 | 0.999 |
| contact-PSNR（只算 `x > 0.12` 的像素） | 整体 PSNR 由 99% 的背景主导，没有意义 | 19.4 → 37.9 dB 单调上升 |
| 输出形状 `[B,128,15,15]` = 28800 | `modeling_vla.py` 的 `deform_proj` 写死了这个维度 | assert 通过 |
| 能被 VLA 加载（`verify_deform_encoder.py`） | `load_deform_encoder_weights` **静默失败**，文件缺失只打一行 warning | missing 0 / unexpected 0 |

最后一条不能省：`scripts/train.py:859` 会把 encoder 冻在它当时持有的任何权重上，
一个随机初始化的冻结 encoder 能把整个 SFT 跑完且不报错。

---

## 3. 与 MoT 架构无关，可跨项目复用

`DeformEncoder` 只 import `torch` / `torchvision`，没有 config、没有 hidden_size、
没有 attention、没有任何 MoT 概念。它就是 `[B,1,H,W] → [B,128,H/16,W/16]` 的纯卷积 backbone。
全部耦合都在 encoder **之外**：

```
modeling_vla.py:108   self.deform_encoder = DeformEncoder()
modeling_vla.py:109   self.deform_proj    = ActionEmbedder(28800, H)   # 128*15*15
modeling_vla.py:508   feats = self.deform_encoder(flat).view(Bd, nf, -1)
```

而且它是全卷积的，**分辨率自由**（实测）：

| 输入 | 输出 | flat |
|---|---|---|
| 240×240 | 128×15×15 | 28800 |
| 224×224 | 128×14×14 | 25088 |
| 128×128 | 128×8×8 | 8192 |
| 320×240 | 128×20×15 | 38400 |

只有 `deform_proj` 把 28800 写死，那是 T-Rex 的约束，不是 encoder 的。
权重名 `sharpa_wave_*` 不是巧合 —— 作者就是在 Sharpa Wave 传感器的形变图上训的，
所以在同款硬件上复用有天然的域优势。

### 3.1 17 个 BatchNorm 是跨项目复用的真正风险

```
stem.1                                      ← 唯一直接看到原始输入尺度的
layer1.0.bn1/bn2   layer1.1.bn1/bn2
layer2.0.bn1/bn2   layer2.0.downsample.1    layer2.1.bn1/bn2
reshape_layer1.1                            ← 论文说的 3×3 re-projection
layer3.0.bn1/bn2   layer3.0.downsample.1    layer3.1.bn1/bn2
reshape_layer2.1                            ← 输出前最后一层
```

`running_mean/var` 没有独立的计算步骤，是**每次 `train()` 模式 forward 的副作用**
（momentum=0.1 的 EMA）。作者的 `num_batches_tracked = 4,226,018` 就是 forward 次数。
这些是**分布统计量**，只在输入分布不变时有效。

> **⚠️ `scripts/train.py:859-862` 只设了 `requires_grad = False`，从未调用 `.eval()`。**
> 对比紧邻的 VQ-VAE（`model.tactile_vqvae.float().eval()`），作者知道这个区别但没对
> deform_encoder 做。而 `requires_grad=False` **拦不住 BN 更新 running stats**（实测：
> `.train()` 下 `running_mean` 改变、`num_batches_tracked` 递增；`.eval()` 下都不变）。
> `train.py:720/935` 都会调 `model.train()`。
>
> 后果：post-train 时这 17 层会在你的数据上自动重校准 —— 好处是免费适配，坏处是
> 所谓"冻结"的 encoder 输出在训练过程中一直在漂，与论文语义不符。要严格冻结，
> 在 862 行后补 `model.deform_encoder.eval()`，并注意 Trainer 每个 epoch 会重新
> `model.train()` 把它打回去（需覆盖 `train()` 或用 callback）。
>
> **在自己的 predictor 里务必显式 `.eval()`。**

### 3.2 需要重估 BN 时怎么做

只需要**你自己传感器的原始形变图** —— 不需要 action、label、teleop 配对、任务成功与否。
量也很小：momentum=0.1 的 EMA 有效窗口约 10 个 batch，一两百个 batch 即收敛。

```python
enc.load_state_dict(...)          # 作者的权重
for m in enc.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.reset_running_stats()   # 清掉 T-Rex 的统计
        m.momentum = None         # 累积移动平均，优于 EMA
enc.train()
with torch.no_grad():
    for x in your_deform_loader:  # [B,1,H,W]，已扣基线并除以 255
        enc(x)
enc.eval()
```

全程 `no_grad`，权重一个都不动，只重写 buffer。

**采样分布必须匹配部署分布。** BN 存的是全局统计量。T-Rex 数据里 thumb 有 26.9% 是全零帧、
index 48.6%、ring 60.4% —— 统计量主要由空气帧主导。校准时如果专挑接触帧喂，
得到的统计量会严重偏离实际，**比不校准更糟**。要么按真实时间顺序采样，要么就别校准。

---

## 4. 自采数据诊断（`check_our_deform.py`）

对 `our_sharpa_teleop/real_test/ep_0000`（task = "pick up the egg"，918 帧）的实测结果。

```bash
python tools/check_our_deform.py \
  --episodes    /path/to/ep_0000 \
  --encoder     /path/to/sharpa_wave_deform_encoder.pth \
  --trex_shards /path/to/deform_frames
```

### [1] 格式 — OK
`tactile/deform` 是 `(918, 5, 240, 240) uint8`，正好是 encoder 要的 240×240，
`deform_valid` 全 True。

### [2] 基线 — 需要改 dataloader

**T-Rex 的形变图背景恰好是 0（82% 的像素），我们的背景是恒定的 2。**

| | frac==0 | frac>30 | std |
|---|---|---|---|
| T-Rex（10 指） | **0.822** | 0.0142 | 16.4 |
| ours/index（原始） | **0.0004** | 0.0149 | 19.4 |
| ours/index（减 2 后） | 0.983 | — | — |

**接触区统计本来就吻合**（`frac>30` 1.49% vs 1.42%，std 19.4 vs 16.4）—— 传感器物理量程
和 T-Rex 一致，差的只有这个地板。dataloader 里加一句：

```python
deform = np.clip(raw.astype(np.int16) - 2, 0, 255).astype(np.uint8) / 255.0
```

脚本是**自动检测众数**得到 floor 的，不是写死 2，标定变了重跑就会报新值。
（值得确认这个 2 是否为 SDK `deform_map_value()` 的量化零点；attrs 的 `units_note`
提到要用它转 mm，可能是我们的 pipeline 少了这步。）

### [3] 活跃度 — 5 指里只有 2 指有足够信号

| finger | uniq 值 | 接触帧 | f6 最大力 |
|---|---|---|---|
| pinky | 1 | 0/918 | 0.006 N |
| ring | 1 | 0/918 | 0.022 N |
| middle | 155 | 9/918 | 1.60 N |
| index | 217 | 261/918 | 17.8 N |
| thumb | 199 | 295/918 | 22.1 N |

pinky/ring 全程只有一个像素值，f6 确认没受力 —— 不是坏了，是"拿鸡蛋"没用到这两根。
918×5 = 4590 张图里真正有信息的约 556 张。

### [4] Hold — deform 实际只有 ~10 Hz，f6 是 30 Hz

```
index   identical-consecutive 0.676  (接触帧内 0.027)   ~10.0 Hz effective
thumb   identical-consecutive 0.685  (接触帧内 0.024)   ~9.7 Hz effective
```

`t_capture` dt 中位数 32.4 ms、`seq` 逐帧递增，但形变图约每 3 帧才真正刷新一次。
与 attrs 里 `tactile_final` 的 `deform_placeholder_frames: 25206 / frames: 25842` 一致 ——
采集线程 97.5% 的装配帧拿到占位符，写盘时被上一帧填充，所以 `deform_valid` 全 True 掩盖了这件事。

括号里是好消息：**真接触时形变几乎每帧都在变**（重复率 2.4-2.7%），hold 集中在空闲段。
静态特征提取不受影响；但若 predictor 要建模形变的**时间演化**，输入序列里 2/3 的
"帧间无变化"是采集端补出来的假象。建议去采集端确认这是 Sharpa 设备刷新率上限还是
UDP 装配丢帧。

### [5] 特征 — **结论：不需要重估 BN，直接冻结用**

减掉基线后，两根有足够接触的手指：

| | dead | 偏差 | std | vs T-Rex |
|---|---|---|---|---|
| T-Rex 参考 | 0.745 | — | 0.309 | — |
| index | 0.729 | −0.016 | 0.462 | ×1.50 |
| thumb | 0.734 | −0.011 | 0.442 | ×1.43 |

dead fraction 差不到 2 个百分点。std 高 1.4-1.5 倍方向合理 —— 这条 episode 全程主动抓取，
接触帧占比（~30%）高于 T-Rex 数据集平均。**判决用 dead fraction 而非 std，就是因为
前者对分布偏移敏感、对接触占比不敏感。**

`middle` 只有 9 帧接触，在 `--feat_stride 20` 下几乎抽不到接触帧，其低 std 是采样假象，
故被 `MIN_CONTACT` 阈值排除在判决之外。

### 附带发现

`tactile/contact_n` 全五指全 918 帧都是 0，而 f6 有 17.8 N / 22.1 N 的力 —— 该字段未被填充。
形变图这条路用不到它，但若 predictor 打算拿它当接触标签，需先查采集端。

---

## 5. 待办

- [ ] dataloader 扣除基线 2（见 §4[2]）
- [ ] 自己的 predictor 里显式 `enc.eval()`（见 §3.1）
- [ ] 查采集端 deform 刷新率 —— 10 Hz vs f6 30 Hz（见 §4[4]）
- [ ] 查 `tactile/contact_n` 为何全 0（见 §4 附带发现）
- [ ] `scripts/train.sh` 的 `DEFORM_ENCODER_PATH` / `VQVAE_CKPT` / `OUTPUT_DIR`
      仍指向作者的 `/mnt/amlfs-02/...`，本地不存在
- [ ] 每采一批新数据重跑 `check_our_deform.py`；判决翻成 "BN stats look stale" 时
      再按 §3.2 重估
