import os
import torch
import argparse
import torch.nn.functional as F
import random
from torch.optim import AdamW
from tqdm import tqdm
from diffusers import DDPMScheduler
from model import DefectFillModel
from data_loader import get_data_loaders
from utils import save_checkpoint, load_checkpoint
# 新增 TensorBoard 支持
from torch.utils.tensorboard import SummaryWriter
import datetime

def train(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建日志文件
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.txt")
    log_file = open(log_file_path, "a")
    log_file.write(f"\n\n{'='*50}\n")
    log_file.write(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Object class: {args.object_class}\n")
    log_file.write(f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
    log_file.write(f"{'='*50}\n\n")
    
    # 创建 TensorBoard 写入器
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Load data
    train_loader, test_loader = get_data_loaders(
        root_dir=args.data_dir,
        object_class=args.object_class,
        batch_size=args.batch_size
    )
    
    # Initialize model
    model = DefectFillModel(
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Set up optimizers with different learning rates for text encoder and UNet
    text_encoder_params = [p for n, p in model.pipeline.text_encoder.named_parameters() if "lora" in n]
    unet_params = [p for n, p in model.pipeline.unet.named_parameters() if "lora" in n]
    
    optimizer = AdamW([
        {"params": text_encoder_params, "lr": args.text_encoder_lr},
        {"params": unet_params, "lr": args.unet_lr}
    ])
    
    # Set up noise scheduler for training
    noise_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", 
        subfolder="scheduler",
        use_safetensors=True  # 添加use_safetensors=True解决安全问题
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, args.resume_from)
        print(f"Resuming from step {start_step}")
        log_file.write(f"Resuming from step {start_step}\n")
    
    # Training loop
    model.pipeline.unet.train()
    model.pipeline.text_encoder.train()
    
    # Total number of training steps
    total_steps = args.max_train_steps
    
    # Progress bar
    progress_bar = tqdm(range(start_step, total_steps), desc="Training")
    
    global_step = start_step
    # 梯度积累计数器
    accumulation_step = 0
    
    # Training loop
    while global_step < total_steps:
        for batch in train_loader:
            # Skip if we've reached max steps
            if global_step >= total_steps:
                break
                
            # Move batch to device
            images = batch["image"].to(device, dtype=torch.float16)
            masks = batch["mask"].to(device, dtype=torch.float16)
            backgrounds = batch["background"].to(device, dtype=torch.float16)
            adjusted_masks = batch["adjusted_mask"].to(device, dtype=torch.float16)
            is_defect = batch["is_defect"]
            
            # 确保我们只处理缺陷样本
            defect_samples = torch.nonzero(is_defect).squeeze(1)
            if len(defect_samples) == 0:
                continue  # 跳过这个批次，如果没有缺陷样本
                
            # 提取所有缺陷样本
            defect_images = images[defect_samples]
            defect_masks = masks[defect_samples]
            defect_backgrounds = backgrounds[defect_samples]
            defect_adjusted_masks = adjusted_masks[defect_samples]
            
            # 获取对象类别和缺陷类型
            object_classes = [batch["object_class"][i] for i in defect_samples]
            
            # 从图片路径中提取缺陷类型
            defect_types = []
            for i in defect_samples:
                if hasattr(train_loader.dataset, 'images') and i < len(train_loader.dataset.images):
                    img_path = train_loader.dataset.images[i]
                    # 假设路径格式为: .../train/defective/defect_type/image.jpg
                    parts = img_path.split(os.sep)
                    for j, part in enumerate(parts):
                        if part == "defective" and j + 1 < len(parts):
                            defect_types.append(parts[j + 1])
                            break
                    else:
                        defect_types.append("defect")  # 默认缺陷类型
                else:
                    defect_types.append("defect")  # 默认缺陷类型
            
            # 学习率预热
            if global_step < args.lr_warmup_steps:
                lr_scale = min(1.0, global_step / args.lr_warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * lr_scale
            
            # 初始化损失
            defect_loss = torch.tensor(0.0, device=device)
            object_loss = torch.tensor(0.0, device=device)
            attention_loss = torch.tensor(0.0, device=device)
            
            # 清除注意力图
            if hasattr(model, 'attention_maps'):
                model.attention_maps = {}
            
            # 处理方式1：使用实际掩码学习缺陷特征 - 对所有缺陷样本
            # 使用具体缺陷类型创建提示
            real_mask_prompts = [f"A photo of {defect_type}" for defect_type in defect_types]
            
            # 编码文本提示
            text_embeddings = model.get_text_embeddings(real_mask_prompts)
            
            # 添加噪声
            noise = torch.randn_like(defect_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (defect_images.shape[0],), device=device)
            noisy_images = noise_scheduler.add_noise(defect_images, noise, timesteps)
            
            # 前向传播
            outputs = model(
                latents=noisy_images,
                timesteps=timesteps,
                encoder_hidden_states=text_embeddings,
                masks=defect_masks
            )
            
            # 计算缺陷损失 (L_def)
            defect_loss = model.get_defect_loss(
                original_images=defect_images,
                masks=defect_masks,
                noisy_latents=noisy_images,
                text_embeddings=text_embeddings,
                timesteps=timesteps
            )
            
            # 获取注意力损失 (L_attn)
            attention_loss = outputs.get("attention_loss", torch.tensor(0.0, device=device))
            
            # 处理方式2：对同一批样本使用随机掩码学习对象完整性
            # 为每个样本生成随机掩码
            random_masks = torch.zeros_like(defect_images[:, :1])
            for i in range(random_masks.shape[0]):
                # 生成随机矩形掩码
                mask = random_masks[i, 0]
                h, w = mask.shape
                
                # 矩形大小在图像尺寸的3%到25%之间
                min_size = int(min(h, w) * 0.03)
                max_size = int(min(h, w) * 0.25)
                
                # 随机矩形尺寸
                rect_h = torch.randint(min_size, max_size, (1,)).item()
                rect_w = torch.randint(min_size, max_size, (1,)).item()
                
                # 随机位置
                y = torch.randint(0, h - rect_h, (1,)).item()
                x = torch.randint(0, w - rect_w, (1,)).item()
                
                # 添加矩形到掩码
                mask[y:y+rect_h, x:x+rect_w] = 1.0
            
            # 创建背景和调整后的掩码
            random_backgrounds = defect_images * (1 - random_masks)
            random_adjusted_masks = random_masks + 0.3 * (1 - random_masks)
            
            # 使用具体对象类别和缺陷类型创建提示
            random_mask_prompts = [f"A {obj_class} with {defect_type}" 
                                   for obj_class, defect_type in zip(object_classes, defect_types)]
            
            # 编码文本提示
            obj_text_embeddings = model.get_text_embeddings(random_mask_prompts)
            
            # 添加噪声
            obj_noise = torch.randn_like(defect_images)
            obj_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (defect_images.shape[0],), device=device)
            obj_noisy_images = noise_scheduler.add_noise(defect_images, obj_noise, obj_timesteps)
            
            # 前向传播
            obj_outputs = model(
                latents=obj_noisy_images,
                timesteps=obj_timesteps,
                encoder_hidden_states=obj_text_embeddings,
                masks=random_masks
            )
            
            # 计算对象损失 (L_obj)
            object_loss = model.get_object_loss(
                original_images=defect_images,
                adjusted_masks=random_adjusted_masks,
                noisy_latents=obj_noisy_images,
                text_embeddings=obj_text_embeddings,
                timesteps=obj_timesteps
            )
            
            # 组合损失，使用权重
            lambda_defect = 0.5  # 缺陷损失权重
            lambda_obj = 0.2     # 对象损失权重
            lambda_attn = 0.05   # 注意力损失权重 - 根据文档设置为0.05
            
            total_loss = lambda_defect * defect_loss + lambda_obj * object_loss + lambda_attn * attention_loss
            
            # 应用梯度积累：将损失除以累积步数
            total_loss = total_loss / args.gradient_accumulation_steps
            
            # 检查是否有NaN值
            if torch.isnan(total_loss):
                print(f"警告: 步骤 {global_step} 中检测到NaN损失")
                print(f"损失值: defect={defect_loss.item()}, object={object_loss.item()}, attention={attention_loss.item()}")
                log_file.write(f"警告: 步骤 {global_step} 中检测到NaN损失\n")
                log_file.write(f"损失值: defect={defect_loss.item()}, object={object_loss.item()}, attention={attention_loss.item()}\n")
                # 跳过这一步的更新
                optimizer.zero_grad()
                continue
            
            # 反向传播
            total_loss.backward()
            
            # 递增梯度积累计数器
            accumulation_step += 1
            
            # 仅在达到指定的梯度积累步数时更新参数
            if accumulation_step >= args.gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                accumulation_step = 0
                
                # 只有在实际更新模型参数时才增加全局步骤和更新进度条
                progress_bar.update(1)
                global_step += 1
                
                # 记录损失
                loss_info = (
                    f"Step: {global_step}/{total_steps}, "
                    f"Defect Loss: {defect_loss.item():.4f}, "
                    f"Object Loss: {object_loss.item():.4f}, "
                    f"Attention Loss: {attention_loss.item():.4f}, "
                    f"Total Loss: {total_loss.item() * args.gradient_accumulation_steps:.4f}"
                )
                
                progress_bar.set_postfix(
                    defect_loss=defect_loss.item(),
                    object_loss=object_loss.item(),
                    attention_loss=attention_loss.item(),
                    total_loss=total_loss.item() * args.gradient_accumulation_steps
                )
                
                # 记录到TensorBoard - 乘以积累步数以获取原始损失值
                writer.add_scalar("Loss/Defect", defect_loss.item(), global_step)
                writer.add_scalar("Loss/Object", object_loss.item(), global_step)
                writer.add_scalar("Loss/Attention", attention_loss.item(), global_step)
                writer.add_scalar("Loss/Total", total_loss.item() * args.gradient_accumulation_steps, global_step)
                
                # 每10步记录学习率
                if global_step % 10 == 0:
                    for i, param_group in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"LearningRate/group{i}", param_group["lr"], global_step)
                
                # 写入日志文件
                log_file.write(f"{loss_info}\n")
                log_file.flush()  # 确保实时写入
                
                # 保存检查点
                if global_step % args.save_steps == 0 or global_step == total_steps:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=global_step,
                        path=os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
                    )
                    log_file.write(f"Checkpoint saved at step {global_step}\n")
            
                # 检查是否达到最大训练步数
                if global_step >= total_steps:
                    break
    
    # 保存最终模型
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=global_step,
        path=os.path.join(args.output_dir, "checkpoint_final.pt")
    )
    
    # 关闭TensorBoard写入器和日志文件
    writer.close()
    log_file.write(f"\nTraining completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.close()
    
    # 返回模型
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DefectFill model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MVTec AD dataset")
    parser.add_argument("--object_class", type=str, required=True, help="Object class to train on")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA adaptation")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha for LoRA adaptation")
    parser.add_argument("--text_encoder_lr", type=float, default=1e-5, help="Learning rate for text encoder")
    parser.add_argument("--unet_lr", type=float, default=1e-5, help="Learning rate for UNet")
    parser.add_argument("--max_train_steps", type=int, default=2000, help="Maximum number of training steps")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # 启动训练
    model = train(args)