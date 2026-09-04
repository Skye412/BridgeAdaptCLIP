# External Evaluation Resolution Protocol Audit

审计日期：2026-09-05  
审计性质：只读链路审计；未修改模型或 evaluator，未训练，未重新运行正式评测。仅执行了 manifest/文件尺寸统计和 3 张样本的前向形状追踪。

## 1. 结论摘要

1. CamCrack789、Crack500、DACL10K 三者都调用同一个 `sliding_window_probability()`。没有任何数据集在进入滑窗前把整张 source image resize 到 1024。
2. CamCrack789 正式 test 的 157 张图全部为 `640×480`；Crack500 正式 test 的 675 张图全部是小于 1024 的预裁 patch，其中 631 张为 `640×360`。二者均采用右/下 replicate padding 到 `1024×1024`，每图只有一个 tile。
3. DACL10K 正式 validation 保持原图尺寸：975 张中 726 张需要多 tile 滑窗，共产生 6396 个 `1024×1024` tile；stride 为 768，预测用非周期二维 Hann 权重拼回原尺寸。
4. 每个 tile 内部有两条不同分辨率路径：
   - CLIP：完整的 `1024×1024` tile（包含 padding 时也包含 padding canvas）bicubic resize 到 `518×518`；
   - Structural：同一个 `1024×1024` tile 直接转 tensor 并归一化，不 resize。
5. tile prediction 始终是 `1024×1024`，滑窗结果裁回 source image 的原始 `H×W`。GT 不 resize；Pixel 指标和 Crack morphology 指标均在 source image 原尺寸上计算。
6. Row0、Fine-v1.3、v2.0、v2.1 使用完全相同的 source manifest、padding、tile starts 和 Hann stitching。结构模型额外拥有 native-1024 structural path，而 Row0 只依赖 518 CLIP path；这是模型架构差异，不是 evaluator 给不同模型使用了不同几何协议。
7. “Crack500 的约 2000×1500 母图被压缩到 1024”不符合本次正式实验事实。服务器上的正式输入已经是 675 个小 patch。图像尺寸仍可能影响结果，但更具体的机制是：小图先 pad 到统一 1024 canvas，使内容在 CLIP 518 输入中占据不同的有效范围；它不能单独解释 structural 模型在 Crack500 上的校准退化。

## 2. 实际调用链

### 2.1 Crack 外部评测

`evaluate_crack_external.py` 的真实链路为：

```text
build_crack_test_manifest()
  -> 打开实际 test_img 文件并读取 EXIF-transposed width/height
  -> 打开 test_lab mask 并强制 image.size == mask.size

逐张 Image.open(source)
  -> ImageOps.exif_transpose
  -> sliding_window_probability(source image)
  -> load_crack_mask()，保持原始尺寸
  -> StreamingBinaryHistogram.update(prediction, GT)
  -> CrackMorphologyMetrics.update(prediction, GT)
```

对应代码：

- `evaluate_crack_external.py:39-63`
- `tools/crack_external.py:13-59`
- `tools/crack_external.py:75-126`

### 2.2 DACL10K 外部评测

`evaluate_dacl10k_external.py` 的真实链路为：

```text
build_validation_manifest()
  -> 读取 official val/img/*.jpg 与 val/ann/*.json
  -> 打开实际图像读取 EXIF-transposed width/height
  -> 强制实际图像尺寸 == annotation['size']

逐张 Image.open(source)
  -> ImageOps.exif_transpose
  -> sliding_window_probability(source image)
  -> rasterize_damage_labels(annotation)，按 annotation 原尺寸栅格化
  -> build_protocol_masks()
  -> ProtocolAccumulator.update(stitched original-size prediction, masks)
```

对应代码：

- `tools/dacl10k_external.py:64-94`
- `tools/dacl10k_external.py:117-152`
- `evaluate_dacl10k_external.py:210-251`

### 2.3 共享 sliding-window 与 tile predictor

共享滑窗的实际行为：

```text
source PIL image
  -> convert RGB（不 resize）
  -> 若 H<1024 或 W<1024，仅在 bottom/right 使用 edge/replicate padding
  -> tile_starts(padded dimension, tile=1024, stride=768)
  -> 每个 tile 为 1024×1024
  -> predictor(tile batch)
  -> 每个 tile 输出连续 1024×1024 probability
  -> non-periodic 2-D Hann weighted average
  -> crop [:source_H, :source_W]
```

`tile_starts()` 会强制加入最右/最下 start，保证完整覆盖。Hann 权重使用 `torch.hann_window(periodic=False)` 的外积，并以 `1e-3` 截断最小值，避免边界权重为零。

对应代码：

- `tools/dacl10k_external.py:47-61`
- `tools/dacl10k_external.py:304-341`

`FrozenTilePredictor` 内部路径：

```text
同一个 1024×1024 tile
  ├─ clip_transform
  │    -> bicubic Resize((518, 518))
  │    -> CenterCrop(518)（此时为恒等裁剪）
  │    -> CLIP/VisualAdapter/TextualAdapter
  │    -> Row0 probability bilinear resize 到 1024×1024
  │
  └─ _structural_tensor
       -> PIL to tensor
       -> /255 + ImageNet normalization
       -> 保持 3×1024×1024

Fine/Broad head -> tile logits/probability 1024×1024
```

对应代码：

- `evaluate_dacl10k_external.py:39-145`
- `tools/utils.py:50-76`
- `tools/bridge_row0.py:29-36`

## 3. 正式 manifest 的实际图像尺寸

所有统计都来自正式结果目录中 evaluator 写出的 manifest。manifest 在生成时实际打开图像读取尺寸，而不是采用数据集论文中描述的母图尺寸。

| 数据集 | 正式图像数 | 唯一尺寸数 | 按面积最小 | 按面积中位样本尺寸 | 按面积最大 | 常见尺寸 |
|---|---:|---:|---:|---:|---:|---|
| CamCrack789 test | 157 | 1 | 640×480 | 640×480 | 640×480 | 640×480：157 |
| Crack500 test | 675 | 3 | 640×360 | 640×360 | 648×484 | 640×360：631；360×640：24；648×484：20 |
| DACL10K-v2 val | 975 | 117 | 448×336 | 1440×1080 | 6000×4000 | 1280×960：143；1024×768：123；1920×1440：80；1600×1200：66；4000×3000：54 |

DACL10K 单独按宽、高统计的范围为：

```text
width  min / marginal median / max = 448 / 1365 / 6000
height min / marginal median / max = 336 / 1080 / 4240
```

## 4. Padding 与 sliding-window 统计

定义：

- `both dimensions <1024`：W<1024 且 H<1024；
- `needs padding`：W<1024 或 H<1024；padding 仅补足不足 1024 的维度；
- `multi tile`：按 padding 后尺寸计算的 tile 数大于 1。

| 数据集 | 两维均<1024 | 需要 padding | 多 tile 图像 | 单 tile 图像 | tile/图 min | tile/图 mean | tile/图 max | 总 tile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CamCrack789 | 157 | 157 | 0 | 157 | 1 | 1.000 | 1 | 157 |
| Crack500 | 675 | 675 | 0 | 675 | 1 | 1.000 | 1 | 675 |
| DACL10K | 95 | 462 | 726 | 249 | 1 | 6.560 | 40 | 6396 |

DACL10K tile-count 分布：

```text
1:249, 2:243, 4:106, 6:136, 8:1, 9:10,
12:46, 15:4, 20:165, 24:9, 30:4, 35:1, 40:1
```

注意：DACL10K 的 `needs padding` 与 `multi tile` 会重叠。例如一边小于 1024、另一边大于 1024 的图像，会在一个方向 padding、另一个方向滑窗。

## 5. 三张代表性样本的实际形状追踪

以下追踪实际实例化了 v2.0 `FrozenTilePredictor` 并执行前向；不是仅由公式推断。

### 5.1 CamCrack789：`image-173`

```text
source image                 480×640
-> replicate padded         1024×1024
-> tile starts              y=[0], x=[0]，1 tile
-> tile                     1024×1024
-> CLIP input               3×518×518
-> structural input         3×1024×1024
-> tile prediction          1024×1024
-> stitched/cropped output  480×640
-> GT                       480×640
```

因此 CamCrack789 不是 `640×480 -> resize 1024`，而是 `640×480 内容 + 右/下复制边界 -> 1024 canvas`。

### 5.2 Crack500：`20160222_080850_1281_721`

```text
source image                 360×640
-> replicate padded         1024×1024
-> tile starts              y=[0], x=[0]，1 tile
-> tile                     1024×1024
-> CLIP input               3×518×518
-> structural input         3×1024×1024
-> tile prediction          1024×1024
-> stitched/cropped output  360×640
-> GT                       360×640
```

正式 `test_img` 中没有约 2000×1500 的输入；675 张均为上述三种预裁尺寸之一。

### 5.3 DACL10K：`dacl10k_v2_validation_0083`

```text
source image                 3000×4000
-> padded                    3000×4000（无需 padding）
-> y starts                  [0, 768, 1536, 1976]
-> x starts                  [0, 768, 1536, 2304, 2976]
-> tiles                     20 × (1024×1024)
-> per-tile CLIP input       3×518×518
-> per-tile structural      3×1024×1024
-> per-tile prediction      1024×1024
-> Hann stitched output     3000×4000
-> rasterized GT            3000×4000
```

## 6. Mask 与指标空间

### CamCrack789

- manifest 构建时逐张检查 image 与 mask 尺寸相同；157/157 无 mismatch；
- mask 在原始尺寸读取，阈值规则为 `uint8 >= 128`；
- 没有 GT resize。

### Crack500

- manifest 构建时逐张检查 image 与 mask 尺寸相同；675/675 无 mismatch；
- mask 在原始尺寸读取，阈值规则为 `binary > 0`；
- 没有 GT resize。

### DACL10K

- manifest 构建时逐张检查实际 image 尺寸等于 annotation `size`；
- polygon 使用 annotation 原始 H/W 栅格化；
- `ProtocolAccumulator.update()` 强制 stitched score、positive mask、ignore mask 三者 shape 相同；
- 没有 GT resize。

### Morphology

Boundary-F1、clDice、Skeleton Recall、Connected-Component Recall 只用于 CamCrack789/Crack500。它们接收裁回 source H/W 的 prediction 和原尺寸 GT；`CrackMorphologyMetrics.update()` 还显式拒绝 shape 不一致。因此这些指标不是在 tile 空间或 518 空间计算。

## 7. 四个模型的几何协议一致性

正式 manifest 的 SHA-256：

```text
CamCrack789 Row0/Fine-v1.3/v2.0/v2.1:
c7eb42f7f980c1fa152ba65711afcfea3f64270ac8ab3e81a8b13db37a7a91ce

Crack500 Row0/Fine-v1.3/v2.0/v2.1:
ef82147d247593c838c6df120573b5eb56b1fdeaa9f008efdda17d09012ee708

DACL10K Row0/Fine-v1.3/Fine-v2.1/v2.0/v2.1:
8dae96bf4a4cf534e9eec8baeb3cd0caeb3ea9ea2e5680494cac52eaa3f0b07a
```

运行脚本对所有模型统一传入：

```text
tile_size=1024
stride=768
tile_batch_size=1（DACL；Crack evaluator 默认也是1）
padding=right/bottom replicate
stitching=non-periodic 2-D Hann weighted probability average
```

因此 source files、顺序、tile starts、padding 和 stitching 不随模型改变。

存在的内部尺度差异是模型设计本身：

| 模型 | CLIP 518 path | Native tile 1024 structural path |
|---|---:|---:|
| Row0 | 是 | 否 |
| Fine-v1.3 | 是 | 是 |
| v2.0 | 是 | 是 |
| v2.1 | 是 | 是 |

Row0 对每个 tile 的信息全部经过 1024→518；结构模型同时看到未 resize 的 1024 tile。这正是结构分支预期提供的尺度优势，不是外部评测不公平。

## 8. DACL edge/center 诊断定义

诊断脚本把每个 tile 外侧 128 像素定义为 local edge。对拼接后的每个 source pixel，分别累计来自所有覆盖 tile 的 Hann-weighted edge contribution 与 center contribution：

```text
edge_dominated   := edge_weight_sum > center_weight_sum
center_dominated := edge_weight_sum <= center_weight_sum
overlap          := coverage_count > 1
non_overlap      := coverage_count == 1
```

因此 edge/center 不是简单按整张图四周切带，也不是“是否处在 overlap”同义词。正式定义见 `tools/dacl10k_external.py:343-407` 和 `diagnose_dacl10k_external.py:145-154`。

## 9. 尺寸是否可能解释 CamCrack789 / Crack500 差异

### 已排除的解释

```text
Crack500 约2000×1500母图
-> 整图压缩到1024
-> 裂缝消失
```

这条链路在正式实验中不存在。实际输入已经是 `640×360 / 360×640 / 648×484` patch。

### 仍然存在的尺寸机制

对于小于 1024 的图片，padding 后的完整 canvas 才送进 CLIP resize：

| 常见输入 | 原内容占 1024 canvas 面积 | 在 518 CLIP canvas 中的近似内容范围 |
|---|---:|---:|
| CamCrack789 640×480 | 29.30% | 约 324×243 |
| Crack500 640×360 | 21.97% | 约 324×182 |

因此 Crack500 的常见 patch 在 CLIP 输入中纵向有效内容更小、复制 padding 比例更高。这是一个真实的尺度/上下文差异，可能影响 Row0 semantic map，并通过 Row0 prior 影响结构模型。

但它不是充分解释：

1. structural branch 仍以原像素密度看到 640×360 内容，没有把其 resize 到 1024；
2. CamCrack789 与 Crack500 都是单 tile、同样的 padding/stitching 代码，却呈现不同的 AP/precision 行为；
3. Crack500 结构模型 Skeleton Recall/Component Recall 上升而 AP/precision 下降，更符合“结构响应存在但外域纹理误报/分数校准较差”，而不是单纯裂缝因缩小而消失。

### 审计后的判断

数据尺寸可能是差异的一个贡献因素，具体是 **padding canvas 改变 CLIP 有效内容尺度**，而不是“母图被压缩到 1024”。当前证据仍要求把域纹理与分数校准视为主要候选原因。若后续做 scale sensitivity，应将其定义为预先固定、全部报告的诊断实验，不能用 Crack500 test 选择最佳 scale。

## 10. 审计证据来源

正式结果目录：

```text
/home/skye/data/Skye/AdaptCLIP/results/dacl10k_external_eval_v1_ff45f61
/home/skye/data/Skye/AdaptCLIP/results/dacl10k_fine_broad_decomposition_d86534c
/home/skye/data/Skye/AdaptCLIP/results/dacl10k_correction_tiling_diagnostics_2250193
/home/skye/data/Skye/AdaptCLIP/results/external_crack_eval_v1_bc721c5
```

本报告没有根据模型结果反向改变任何预处理参数，也没有生成新的正式指标。
