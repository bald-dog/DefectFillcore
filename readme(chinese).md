
# DefectFill2 非官方实现

基于 Stable Diffusion Inpainting 思想的工业缺陷生成与填补实验项目。  
DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection  
arXiv:https://arxiv.org/abs/2503.13985

---

## 1. 主要功能
- 核心模型封装：[*model.py*](model.py)
- 训练脚本（包含注意力损失、梯度累积、TensorBoard）：[*train.py*](train.py)

- 推理 / 生成脚本：[*inference.py*](inference.py)
- 数据加载与预处理：[*data_loader.py*](data_loader.py)
- 工具函数（保存/加载权重等）：[*utils.py*](utils.py)
---

## 2. 环境准备

### 2.1 Python 版本
建议使用 Python 3.11（笔记本环境显示 3.11.9）。

### 2.2 安装依赖
项目依赖列于 [requirements.txt](requirements.txt)，执行：
```bash
pip install -r requirements.txt
```
主要涉及：
- torch / torchvision
- diffusers
- transformers
- tqdm
- tensorboard
### 2.3 推荐 Conda 环境
```bash
conda create -n defectfill2 python=3.11 -y
conda activate defectfill2
pip install -r requirements.txt
```

---

## 3. 数据集


### 3.1 目录结构示例
```
/home/phpc/program2/DefectFill2/MVTec2/
  bottle/
    train/defective/broken_large/000.png ...
    train/defective_masks/broken_large/000_mask.png ...
    000.png ...
```
### 3.2 数据集下载  
我们将调整适合本项目目录结构的MVTec数据集上传到了百度网盘。
链接: https://pan.baidu.com/s/1xwr5AkrmLi6ahPvU5eQL8Q 提取码: g84e 
--来自百度网盘超级会员v7的分享
---

## 4. 训练数据管线

来自 [train.py](train.py) ：
- 每批样本字段（示例）：`image`, `mask`, `background`, `adjusted_mask`, `is_defect`, `object_class`, `defect_type`
- 仅处理含缺陷样本：筛选 `is_defect == 1`
- 两类路径（同一缺陷样本双重使用，1:1 逻辑）：
  1. 使用真实缺陷掩码：学习缺陷语义（Defect Loss）
     - Prompt 模板：`"A photo of {defect_type}"`
  2. 使用随机矩形掩码：学习对象完整性（Object Loss）
     - Prompt 模板：`"A {object_class} with {defect_type}"`

随机矩形掩码生成逻辑见：
- [train.py](train.py) 第 193 行附近
---

## 5. 损失函数

总损失（示例，见 [train.py](train.py) 第 235 行附近）：
```
L_total = λ_defect * L_defect + λ_object * L_object + λ_attn * L_attention
```
---

## 6. 训练

### 6.1 基本命令
```bash
python train.py \
  --data_dir "/home/phpc/program2/DefectFill2/MVTec" \
  --object_class bottle \
  --output_dir "./models/bottle" \
  --batch_size 2 \
  --max_train_steps 2000
```

可选参数（参见 [train.py](train.py) ArgumentParser）：
- `--lora_rank`
- `--learning_rate`
- `--save_steps`
- `--resume_from`
- `--gradient_accumulation_steps`
- `--lr_warmup_steps`
- `--seed`

### 6.2 训练特性
- 梯度累积：见 [train.py](train.py) 第 247 行附近
- 动态学习率 warmup：见 [train.py](train.py) 第 131 行附近
- Checkpoint 保存：每 `save_steps` 或结束时，调用 [`utils.save_checkpoint`](utils.py)
- 日志/监控：
  - 文本日志：`train.txt`
  - TensorBoard：`output_dir/logs`  
    启动：
    ```bash
    tensorboard --logdir ./models/bottle/logs --port 6006
    ```

---

## 7. 推理 / 生成

脚本：[inference.py](inference.py)

示例（来自需求记录）：
```bash
python inference.py \
  --checkpoint checkpoint_2000.pt \
  --object_class bottle \
  --image_path 000.png \
  --mask_path 012_mask.png \
  --defect_type broken_large
```

输出：
- 生成缺陷填充 / 合成图像
- 可扩展保存中间注意力图（需在模型中启用 `attention_maps`）

---

## 8. 模型与可微调组件

在 [model.py](model.py) 中封装 `DefectFillModel`（名称来自引用），加载 Stable Diffusion Inpainting 相关组件（UNet + 文本编码器 + VAE）。  
微调策略（根据代码推测）：
- LoRA（`--lora_rank`）
- 仅对指定层参数解冻（需结合实际实现查看）

---

## 9. 注意事项

1. 输入通道：Stable Diffusion Inpainting UNet 典型输入 9 通道（latent 4 + masked latent 4 + mask 1）；请确保前处理与模型封装一致（若需核实，请检查 [model.py](model.py) 内 forward / pipeline 封装逻辑）。
2. 当前训练优化目标为“噪声预测”形式（DDPM 典型 MSE），损失基于模型输出与真实噪声。
3. 若需改为“重建图像空间优化”，需在采样链末端添加重建与像素 / 感知损失。
4. 注意力图缓存清空位置见：
   - [train.py](train.py) 第 131 行附近：`model.attention_maps = {}`

---

## 10. 常见问题 (FAQ)

| 问题 | 说明 |
| ---- | ---- |
| 训练显存不足 | 降低 `batch_size` 或使用更小 `lora_rank` |
| Loss 出现 NaN | 已在代码中检测（见 [train.py](train.py) 第 235/247 段），出现时跳过更新 |
| 注意力图不收敛 | 调整 `λ_attn`（如从 0.05 降至 0.02） |
| 推理缺陷位置偏移 | 检查 mask 与原图尺寸/坐标是否对齐 |

---

## 11. 扩展方向

- 增加多尺度注意力一致性（当前已做层平均）
- 引入掩码引导的 latent blending
- 加入感知损失（LPIPS）提升视觉质量
- 支持多缺陷联合提示：`"A bottle with broken_large and contamination"`

---

## 12. 快速上手步骤

```bash
# 1. 创建环境
conda create -n defectfill2 python=3.11 -y
conda activate defectfill2

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动训练
python train.py --data_dir ./MVTec2 --object_class bottle --output_dir ./models/bottle

# 4. 监控
tensorboard --logdir ./models/bottle/logs

# 5. 推理
python inference.py --checkpoint checkpoint_2000.pt \
  --object_class bottle \
  --image_path ./some_input.png \
  --mask_path ./some_mask.png \
  --defect_type broken_large
```

---

## 13. 参考文件索引

- 训练主循环: [train.py](train.py)
- 模型封装: [model.py](model.py)
- 数据加载: [data_loader.py](data_loader.py)
- 推理脚本: [inference.py](inference.py)
- Checkpoint 工具: [utils.py](utils.py)
- 训练日志样例: [train.txt](train.txt)



