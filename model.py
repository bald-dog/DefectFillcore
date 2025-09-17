import torch
import torch.nn as nn
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel
from peft import LoraConfig, get_peft_model
import lpips
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
from diffusers.models.attention_processor import Attention, AttnProcessor


class AttentionStoreProcessor(AttnProcessor):
    """注意力处理器，用于存储交叉注意力图"""
    def __init__(self, model=None):
        super().__init__()
        self.model = model  # 存储对模型的引用
    
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)
        
        is_cross_attention = encoder_hidden_states is not None
        
        if not is_cross_attention:
            # 对于自注意力，直接使用标准处理
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            # 对于交叉注意力，我们需要存储注意力图
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * attn.scale
        
        # 应用softmax得到注意力概率
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # 存储交叉注意力概率到模型实例中
        if is_cross_attention and self.model is not None:
            # 获取当前块的名称
            for name, module in self.model.pipeline.unet.named_modules():
                if module == attn and "attn2" in name and "up_blocks" in name:
                    try:
                        # 提取批次大小和头数
                        num_heads = attn.heads
                        
                        # 获取注意力概率的总元素数
                        total_elements = attention_probs.numel()
                        
                        # 计算query_tokens
                        query_len = hidden_states.shape[1]
                        
                        # 计算key_tokens (通常是77，与文本编码器输出长度相关)
                        key_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else query_len
                        
                        # 安全地重塑 - 检查形状是否兼容
                        expected_size = batch_size * num_heads * query_len * key_len
                        if total_elements == expected_size:
                            reshaped_probs = attention_probs.reshape(
                                batch_size, 
                                num_heads, 
                                query_len,
                                key_len
                            )
                            if not hasattr(self.model, "attention_maps"):
                                self.model.attention_maps = {}
                            self.model.attention_maps[name] = reshaped_probs.detach().clone()
                        else:
                            print(f"警告: 注意力概率形状不兼容 - 总元素: {total_elements}, 预期: {expected_size}")
                    except Exception as e:
                        print(f"注意力图处理错误: {e}, 形状: {attention_probs.shape}")
                    break
        
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states
class DefectFillModel(nn.Module):
    def __init__(self, device="cuda", lora_rank=8, lora_alpha=32, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        
        # 加载模型
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16
        ).to(device)
        
        # 固定随机种子以确保可重复性
        self.pipeline.set_progress_bar_config(disable=True)
        
        # 使用DDIM调度器进行推理
        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            subfolder="scheduler"
        )
        self.scheduler.set_timesteps(30)
        
        # LoRA配置
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights="gaussian"
        )
        
        text_encoder_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            init_lora_weights="gaussian"
        )
        
        # 应用LoRA适配器
        self.pipeline.unet = get_peft_model(self.pipeline.unet, unet_lora_config)
        self.pipeline.text_encoder = get_peft_model(self.pipeline.text_encoder, text_encoder_lora_config)
        
        # 冻结VAE的参数
        for param in self.pipeline.vae.parameters():
            param.requires_grad = False
        
        # 用于计算LPIPS损失的VGG模型
        self.lpips_model = lpips.LPIPS(net='vgg').to(device)
        
        # 初始化注意力图存储
        self.attention_maps = {}
        
        # 注册自定义注意力处理器
        self.register_attention_processor()
        
        # 存储缺陷词元的索引
        self.defect_token_idx = None
    def register_attention_processor(self):
        """替换UNet中的注意力处理器为我们的自定义处理器"""
        # 清空现有的注意力图
        self.attention_maps = {}
        
        # 替换注意力处理器
        for name, module in self.pipeline.unet.named_modules():
            if isinstance(module, Attention) and "attn2" in name:  # 只处理交叉注意力
                module.processor = AttentionStoreProcessor(model=self)

    def get_attention_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """
        计算注意力损失 - 强制缺陷词元的注意力图与缺陷掩码对齐
        """
        # 如果没有收集到注意力图，返回零损失
        if not self.attention_maps:
            return torch.tensor(0.0, device=masks.device)
        
        batch_size = masks.shape[0]
        attention_loss = torch.tensor(0.0, device=masks.device)
        
        # 筛选解码器层的注意力图
        decoder_attention_maps = {
            name: attn_map for name, attn_map in self.attention_maps.items()
            if "up_blocks" in name  # 只使用解码器层的注意力图
        }
        
        if not decoder_attention_maps:
            return torch.tensor(0.0, device=masks.device)
        
        # 使用之前找到的缺陷词元索引
        defect_token_idx = self.defect_token_idx if self.defect_token_idx is not None else 3
        
        # 对每个批次样本计算注意力损失
        for b in range(batch_size):
            mask = masks[b].squeeze(0)  # (H, W)
            
            # 获取当前批次的所有解码器层注意力图并调整大小
            resized_attention_maps = []
            for name, attn_map in decoder_attention_maps.items():
                try:
                    # 提取当前批次样本的缺陷词元注意力图
                    if b < attn_map.shape[0]:  # 确保批次索引有效
                        # 所有注意力头对特定词元的注意力，然后求平均
                        defect_attn = attn_map[b, :, :, defect_token_idx].mean(dim=0)
                        
                        # 将注意力图重塑为2D图像
                        seq_len = defect_attn.shape[0]
                        h = int(math.sqrt(seq_len))
                        if h * h == seq_len:  # 确保能形成一个完美的正方形
                            defect_attn = defect_attn.reshape(h, h)
                            
                            # 调整注意力图大小以匹配掩码尺寸
                            resized_attn = F.interpolate(
                                defect_attn.unsqueeze(0).unsqueeze(0),
                                size=mask.shape,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                            
                            resized_attention_maps.append(resized_attn)
                except Exception as e:
                    print(f"注意力图处理错误: {e}")
                    continue
            
            # 计算所有解码器层注意力图的平均值
            if resized_attention_maps:
                avg_attn_map = torch.stack(resized_attention_maps).mean(dim=0)
                
                # 计算L2损失：||A_t^[V*] - M||_2^2
                sample_loss = F.mse_loss(avg_attn_map, mask)
                attention_loss += sample_loss
        
        # 返回批次平均损失
        return attention_loss / batch_size if batch_size > 0 else attention_loss


    def get_text_embeddings(self, prompts):
        """
        将文本提示编码为嵌入向量，并找到缺陷词元的索引
        
        Args:
            prompts: 文本提示列表
        
        Returns:
            编码后的文本嵌入
        """
        # 确保pipeline准备好
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            raise ValueError("Pipeline not initialized")
            
        # 确保prompts是列表
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # 寻找缺陷词元的索引位置
        # 这里我们假设defect_type出现在prompt中，通常在"A photo of {defect_type}"或"A {obj_class} with {defect_type}"
        sample_prompt = prompts[0]
        tokens = self.pipeline.tokenizer.tokenize(sample_prompt)
        
        # 查找可能的缺陷词元位置
        # 通常defect_type会在"photo of"或"with"之后出现
        defect_keywords = ["photo of", "with"]
        for keyword in defect_keywords:
            if keyword in sample_prompt.lower():
                # 在词元列表中找到keyword后的位置
                for i, token in enumerate(tokens[:-1]):  # 排除最后一个词元
                    if keyword.split()[-1] in token.lower():  # 匹配keyword的最后一个词
                        # 假设缺陷词元是keyword后的第一个词元
                        self.defect_token_idx = i + 2  # +1跳过keyword，+1跳到下一个词元
                        break
        
        # 如果没有找到合适的位置，默认使用第4个词元（基于经验值）
        if self.defect_token_idx is None:
            self.defect_token_idx = 3
            
        # 获取文本嵌入
        text_inputs = self.pipeline.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.pipeline.device)
        
        with torch.no_grad():
            text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids)[0]
            
        return text_embeddings
    
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        """
        # 每次前向传播前清空注意力图
        self.attention_maps = {}
        
        # 确保掩码有正确的形状
        if len(masks.shape) == 3:  # [batch_size, height, width]
            masks = masks.unsqueeze(1)
        elif len(masks.shape) == 2:  # [height, width]
            masks = masks.unsqueeze(0).unsqueeze(0)
        
        # 编码到潜在空间
        with torch.no_grad():
            latents = self.pipeline.vae.encode(latents).latent_dist.sample() * self.pipeline.vae.config.scaling_factor
        
        # 将掩码调整为潜在空间大小
        mask_latents = F.interpolate(masks, size=(latents.shape[2], latents.shape[3]))
        
        # 创建掩码潜在表示
        masked_latents = latents * (1 - mask_latents)
        
        # 组成9通道输入
        concat_latents = torch.cat([latents, masked_latents, mask_latents], dim=1)
        
        # 前向传播 - 这会触发注意力处理器收集注意力图
        noise_pred = self.pipeline.unet(
            concat_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        
        # 计算注意力损失 - 现在应该能够获取到注意力图
        attention_loss = self.get_attention_loss(masks)
        
        # 打印调试信息
        #print(f"注意力损失: {attention_loss.item()}")
        
        return {
            "noise_pred": noise_pred,
            "mask_latents": mask_latents,
            "latents": latents,
            "attention_loss": attention_loss
        }
    
    def denoise_latents(self, noisy_latents, text_embeddings, masks, timesteps, num_inference_steps=10):
        """
        将噪声潜变量转换为最终生成的图像
        
        Args:
            noisy_latents: 带噪声的潜变量
            text_embeddings: 文本嵌入
            masks: 掩码
            timesteps: 起始时间步
            num_inference_steps: 推理步数
        
        Returns:
            生成的图像
        """
        batch_size = noisy_latents.shape[0]
        device = noisy_latents.device
        
        # 配置调度器
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 将调度器的时间步移到正确的设备上
        scheduler_timesteps = self.scheduler.timesteps.to(device)
        
        # 确保timesteps在正确的设备上
        if isinstance(timesteps, torch.Tensor):
            timesteps = timesteps.to(device)
        else:
            timesteps = torch.tensor(timesteps, device=device)
        
        # 获取与给定时间步对应的调度器时间步索引
        step_indices = []
        for t in timesteps:
            # 将单个时间步转换为标量
            t_item = t.item() if isinstance(t, torch.Tensor) else t
            # 检查t是否在scheduler_timesteps中
            matches = (scheduler_timesteps == t_item).nonzero()
            if len(matches) > 0:
                step_indices.append(matches.item())
            else:
                step_indices.append(0)
        
        # 确保掩码有正确的形状
        if len(masks.shape) == 3:  # [batch_size, height, width]
            masks = masks.unsqueeze(1)
        elif len(masks.shape) == 2:  # [height, width]
            masks = masks.unsqueeze(0).unsqueeze(0)
        
        # 将掩码调整为潜在空间大小
        mask_latents = F.interpolate(masks, size=(noisy_latents.shape[2], noisy_latents.shape[3]))
        
        # 存储中间结果
        latents = noisy_latents.clone()
        
        # 从当前时间步开始去噪
        for i, t in enumerate(scheduler_timesteps):
            # 跳过之前的步骤
            if i < min(step_indices):
                continue
                
            # 准备当前批次中每个样本的时间步
            batch_timesteps = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
            
            # 创建掩码潜在表示
            masked_latents = latents * (1 - mask_latents)
            
            # 将潜变量和掩码连接起来
            concat_latents = torch.cat([latents, masked_latents, mask_latents], dim=1)
            
            # 预测噪声
            with torch.enable_grad():  # 确保梯度计算
                noise_pred = self.pipeline.unet(
                    concat_latents,
                    batch_timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # 应用调度器步骤以更新潜变量
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 将最终的潜变量解码为图像
        latents = 1 / self.pipeline.vae.config.scaling_factor * latents
        
        # 解码生成的图像
        with torch.enable_grad():  # 确保梯度计算
            images = self.pipeline.vae.decode(latents).sample
        
        # 从[-1, 1]范围转换为[0, 1]范围
        images = (images + 1) / 2
        
        return images
    
    def get_defect_loss(
        self,
        original_images: torch.Tensor,
        masks: torch.Tensor,
        noisy_latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算缺陷损失 (L_def) - 基于噪声预测的MSE损失
        """
        # 获取批次大小
        batch_size = noisy_latents.shape[0]
        
        # 将原始图像编码到潜在空间 - 分离此操作避免不必要的梯度计算
        with torch.no_grad():
            original_latents = self.pipeline.vae.encode(original_images).latent_dist.sample() * self.pipeline.vae.config.scaling_factor
        
        # 获取噪声调度器
        noise_scheduler = self.pipeline.scheduler
        
        # 添加噪声到原始潜在向量 - 这部分需要梯度
        noise = torch.randn_like(original_latents)
        noisy_original_latents = noise_scheduler.add_noise(original_latents, noise, timesteps)
        
        # 确保掩码尺寸正确
        if len(masks.shape) == 3:  # [batch_size, height, width]
            masks = masks.unsqueeze(1)
        
        # 将掩码调整为潜在空间大小
        mask_latents = F.interpolate(masks, size=(original_latents.shape[2], original_latents.shape[3]))
        
        # 创建掩码潜在表示（将掩码区域置为0）
        masked_latents = noisy_original_latents * (1 - mask_latents)
        
        # 将三个表示连接成9通道输入 (4+4+1)
        concat_latents = torch.cat([noisy_original_latents, masked_latents, mask_latents], dim=1)
        
        # 使用UNet预测噪声 - 确保这部分计算梯度
        noise_pred = self.pipeline.unet(
            concat_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
        ).sample
        
        # 只在掩码区域计算损失
        weighted_loss = mask_latents * ((noise_pred - noise) ** 2)
        
        # 计算非零元素的平均值
        return torch.sum(weighted_loss) / torch.sum(mask_latents + 1e-8)

    def get_object_loss(
        self,
        original_images: torch.Tensor,
        adjusted_masks: torch.Tensor,
        noisy_latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算对象损失 (L_obj) - 基于噪声预测的MSE损失，带有调整权重
        """
        # 获取批次大小
        batch_size = noisy_latents.shape[0]
        
        # 将原始图像编码到潜在空间 - 分离此操作避免不必要的梯度计算
        with torch.no_grad():
            original_latents = self.pipeline.vae.encode(original_images).latent_dist.sample() * self.pipeline.vae.config.scaling_factor
        
        # 获取噪声调度器
        noise_scheduler = self.pipeline.scheduler
        
        # 添加噪声到原始潜在向量 - 这部分需要梯度
        noise = torch.randn_like(original_latents)
        noisy_original_latents = noise_scheduler.add_noise(original_latents, noise, timesteps)
        
        # 确保掩码有正确的形状
        if len(adjusted_masks.shape) == 3:  # [batch_size, height, width]
            adjusted_masks = adjusted_masks.unsqueeze(1)
        
        # 将掩码调整为潜在空间大小
        mask_latents = F.interpolate(adjusted_masks, size=(original_latents.shape[2], original_latents.shape[3]))
        
        # 创建掩码潜在表示
        masked_latents = noisy_original_latents * (1 - mask_latents)
        
        # 将三个表示连接成9通道输入 (4+4+1)
        concat_latents = torch.cat([noisy_original_latents, masked_latents, mask_latents], dim=1)
        
        # 使用UNet预测噪声 - 确保这部分计算梯度
        noise_pred = self.pipeline.unet(
            concat_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
        ).sample
        
        # 使用调整的掩码计算损失
        weighted_loss = mask_latents * ((noise_pred - noise) ** 2)
        
        # 计算非零元素的平均值
        return torch.sum(weighted_loss) / torch.sum(mask_latents + 1e-8)
        
    def get_attention_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """计算注意力损失 - 强制缺陷词元的注意力图与缺陷掩码对齐"""
        # 如果没有收集到注意力图，返回零损失
        if not hasattr(self, 'attention_maps') or not self.attention_maps:
            print("警告: 没有找到注意力图!")
            return torch.tensor(0.0, device=masks.device)
        
        batch_size = masks.shape[0]
        attention_loss = torch.tensor(0.0, device=masks.device)
        
        # 筛选解码器层的注意力图
        decoder_attention_maps = {
            name: attn_map for name, attn_map in self.attention_maps.items()
            if "up_blocks" in name  # 只使用解码器层的注意力图
        }
        
        if not decoder_attention_maps:
            print("警告: 没有找到解码器层注意力图!")
            return torch.tensor(0.0, device=masks.device)
        
        # 调试信息
        #print(f"找到的解码器注意力图: {list(decoder_attention_maps.keys())}")
        
        # 使用之前找到的缺陷词元索引
        defect_token_idx = self.defect_token_idx if self.defect_token_idx is not None else 3
        
        # 对每个批次样本计算注意力损失
        for b in range(batch_size):
            mask = masks[b].squeeze(0)  # (H, W)
            
            # 获取当前批次的所有解码器层注意力图并调整大小
            resized_attention_maps = []
            for name, attn_map in decoder_attention_maps.items():
                try:
                    # 提取当前批次样本的缺陷词元注意力图
                    if b < attn_map.shape[0]:  # 确保批次索引有效
                        # 所有注意力头对特定词元的注意力，然后求平均
                        defect_attn = attn_map[b, :, :, defect_token_idx].mean(dim=0)
                        
                        # 将注意力图重塑为2D图像
                        seq_len = defect_attn.shape[0]
                        h = int(math.sqrt(seq_len))
                        if h * h == seq_len:  # 确保能形成一个完美的正方形
                            defect_attn = defect_attn.reshape(h, h)
                            
                            # 调整注意力图大小以匹配掩码尺寸
                            resized_attn = F.interpolate(
                                defect_attn.unsqueeze(0).unsqueeze(0),
                                size=mask.shape,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                            
                            resized_attention_maps.append(resized_attn)
                except Exception as e:
                    print(f"注意力图处理错误: {e}, 形状: {attn_map.shape}")
                    continue
            
            # 计算所有解码器层注意力图的平均值
            if resized_attention_maps:
                avg_attn_map = torch.stack(resized_attention_maps).mean(dim=0)
                
                # 计算L2损失：||A_t^[V*] - M||_2^2
                sample_loss = F.mse_loss(avg_attn_map, mask)
                attention_loss += sample_loss
            else:
                print(f"警告: 批次 {b} 未能处理任何注意力图")
        
        # 返回批次平均损失
        if batch_size > 0:
            return attention_loss / batch_size
        else:
            return attention_loss
        
    def inference(self, clean_image, mask, prompt, num_samples=1, steps=30, guidance_scale=7.5):
        """
        对一个图像执行推理，生成缺陷区域
        
        Args:
            clean_image: 干净的原始图像张量 [1, 3, H, W]
            mask: 缺陷掩码张量 [1, 1, H, W]
            prompt: 文本提示
            num_samples: 要生成的样本数量
            steps: 推理步数
            guidance_scale: 分类器自由引导尺度
            
        Returns:
            生成的图像张量 [1, 3, H, W]
        """
        # 确保图像和掩码具有正确的形状
        if len(clean_image.shape) == 3:
            clean_image = clean_image.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)
            
        # 确保图像在[-1, 1]范围内
        if clean_image.min() >= 0 and clean_image.max() <= 1:
            clean_image = 2 * clean_image - 1
            
        # 确保掩码在[0, 1]范围内
        if mask.max() > 1:
            mask = mask / 255.0
            
        # 编码文本提示
        text_embeddings = self.get_text_embeddings([prompt])
        
        # 将图像编码到潜在空间
        with torch.no_grad():
            latents = self.pipeline.vae.encode(clean_image).latent_dist.sample() * self.pipeline.vae.config.scaling_factor
            
        # 使用模型进行去噪处理
        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps
        
        # 根据调度器的噪声级别添加噪声
        noisy_latents = torch.randn_like(latents)
        
        # 去噪生成图像
        generated_images = self.denoise_latents(
            noisy_latents=noisy_latents,
            text_embeddings=text_embeddings,
            masks=mask,
            timesteps=timesteps[-1].repeat(latents.shape[0]),
            num_inference_steps=steps
        )
        
        return generated_images