import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import DefectFillModel
from utils import load_checkpoint
from torchvision.utils import save_image
from torchvision import transforms

def inference(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = DefectFillModel(
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Load checkpoint
    if args.checkpoint:
        load_checkpoint(model, None, args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Set to eval mode
    model.pipeline.unet.eval()
    model.pipeline.text_encoder.eval()
    
    # 添加修复函数，增加defect_type参数
    def fixed_inference(model, clean_image, mask, object_class, defect_type, num_samples=8, steps=50, guidance_scale=7.5):
        # 创建包含具体缺陷类型的prompt
        prompt = f"A {object_class} with {defect_type}"
        print(f"Using prompt: '{prompt}'")
        
        # 生成多个样本
        samples = []
        best_score = -1
        best_sample = None
        
        # 确保clean_image和mask_tensor已经在设备上且形状正确
        clean_tensor = clean_image.to(device)
        mask_tensor = mask.to(device)
        
        # 记录输入图像的尺寸
        _, _, h_input, w_input = clean_tensor.shape
        
        # 对于pipeline，需要将图像从[-1,1]转换到[0,1]范围
        # 我们在这里存储模型的训练格式图像，以便LPIPS计算使用
        clean_tensor_model_format = clean_tensor.clone()
        
        # 转换图像到[0,1]范围给pipeline使用
        clean_tensor_pipeline = (clean_tensor + 1) / 2.0
        
        for i in range(num_samples):
            # 为不同的样本设置不同的种子
            generator = torch.Generator(device=device).manual_seed(i)
            
            # 调用pipeline生成图像，显示进度条
            result = model.pipeline(
                prompt=prompt,
                image=clean_tensor_pipeline,
                mask_image=mask_tensor,
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                callback_on_step_end=lambda pipeline, step, timestep, callback_kwargs: (print(f"Step {step}/{steps}", end="\r") if step % 10 == 0 else None) or {}
            )
            # 处理PIL Image
            sample_image = result.images[0]
            
            # 转换为张量并添加批处理维度
            sample_tensor = transforms.ToTensor()(sample_image).unsqueeze(0)
            
            # 转换到模型使用的[-1,1]范围用于LPIPS计算
            sample_tensor_model_format = (sample_tensor * 2.0) - 1.0
            
            # 移到正确的设备
            sample_tensor_model_format = sample_tensor_model_format.to(device)
            
            # 确保尺寸正确
            if sample_tensor_model_format.shape[-2:] != (h_input, w_input):
                sample_tensor_model_format = torch.nn.functional.interpolate(
                    sample_tensor_model_format, 
                    size=(h_input, w_input),
                    mode='bilinear',
                    align_corners=False
                )
                
            samples.append(sample_tensor_model_format)
            print(f"Generated sample {i+1}/{num_samples}")
        
        # 使用LPIPS评估样本
        for idx, sample in enumerate(samples):
            # 确保掩码与样本具有相同的空间尺寸
            if mask_tensor.shape[-2:] != sample.shape[-2:]:
                mask_resized = torch.nn.functional.interpolate(
                    mask_tensor, 
                    size=sample.shape[-2:],
                    mode='nearest'
                )
            else:
                mask_resized = mask_tensor
            
            # 只在掩码区域计算LPIPS
            lpips_score = model.lpips_model(
                clean_tensor_model_format * mask_resized,
                sample * mask_resized
            ).mean()
            
            print(f"Sample {idx+1} LPIPS score: {lpips_score.item():.4f}")
            
            if lpips_score > best_score:
                best_score = lpips_score
                best_sample = sample
        
        print(f"Best sample selected with LPIPS score: {best_score.item():.4f}")
        # 返回最佳样本，去掉批处理维度
        return best_sample.squeeze(0)
    
    # Transformations - 仍然使用[-1,1]范围的标准化，但在pipeline调用前会转换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    if args.image_dir:
        # Process entire directory
        image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in tqdm(image_files, desc="Processing images"):
            # 这里显示当前处理的图像
            print(f"\nProcessing image: {image_file}")
            
            image_path = os.path.join(args.image_dir, image_file)
            mask_path = os.path.join(args.mask_dir, image_file.replace('.jpg', '.png').replace('.jpeg', '.png')) if args.mask_dir else None
            
            # Skip if mask doesn't exist
            if mask_path and not os.path.exists(mask_path):
                print(f"Mask not found for {image_file}, skipping")
                continue
            
            # Load image
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Load or generate mask
            if mask_path:
                mask = Image.open(mask_path).convert("L")
                mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(device)
                
                # 从掩码路径中提取缺陷类型
                defect_type = "sks"  # 默认值
                # 提取掩码路径中的缺陷类型文件夹名
                mask_path_parts = mask_path.split(os.sep)
                for i, part in enumerate(mask_path_parts):
                    if part == "defective_masks" and i+1 < len(mask_path_parts):
                        defect_type = mask_path_parts[i+1]
                        break
            else:
                # Generate random mask if none provided
                mask_tensor = torch.zeros((1, 1, 256, 256), device=device)
                num_rectangles = 5
                
                for _ in range(num_rectangles):
                    # Rectangle size between 5% and 20% of image dimensions
                    min_size = int(256 * 0.05)
                    max_size = int(256 * 0.2)
                    
                    # Random rectangle dimensions
                    rect_h = np.random.randint(min_size, max_size)
                    rect_w = np.random.randint(min_size, max_size)
                    
                    # Random position
                    y = np.random.randint(0, 256 - rect_h)
                    x = np.random.randint(0, 256 - rect_w)
                    
                    # Add rectangle to mask
                    mask_tensor[0, 0, y:y+rect_h, x:x+rect_w] = 1.0
                defect_type = args.defect_type if args.defect_type else "sks"
            
            # 如果提供了命令行参数的缺陷类型，优先使用
            if args.defect_type:
                defect_type = args.defect_type
                
            # Generate defect image
            with torch.no_grad():
                # 使用修改后的推理函数，传递缺陷类型
                defect_img = fixed_inference(
                    model=model,
                    clean_image=img_tensor,
                    mask=mask_tensor,
                    object_class=args.object_class,
                    defect_type=defect_type,  # 传递缺陷类型
                    num_samples=args.num_samples,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale
                )
            
            # Save generated image
            output_path = os.path.join(args.output_dir, f"defect_{image_file}")
            save_image((defect_img + 1) / 2, output_path)
            
            # Also save the mask and original for comparison
            save_image(mask_tensor, os.path.join(args.output_dir, f"mask_{image_file}"))
            save_image((img_tensor + 1) / 2, os.path.join(args.output_dir, f"original_{image_file}"))
    
    elif args.image_path:
        # 显示当前处理的图像
        print(f"\nProcessing single image: {args.image_path}")
        
        img = Image.open(args.image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 确定缺陷类型
        defect_type = args.defect_type if args.defect_type else "sks"  # 使用参数或默认值
        
        # Load or generate mask
        if args.mask_path:
            mask = Image.open(args.mask_path).convert("L")
            mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(device)
            
            # 如果未提供缺陷类型参数，尝试从掩码路径提取
            if not args.defect_type:
                mask_path_parts = args.mask_path.split(os.sep)
                for i, part in enumerate(mask_path_parts):
                    if part == "defective_masks" and i+1 < len(mask_path_parts):
                        defect_type = mask_path_parts[i+1]
                        break
        else:
            # Generate random mask if none provided
            mask_tensor = torch.zeros((1, 1, 256, 256), device=device)
            num_rectangles = 5
            
            for _ in range(num_rectangles):
                # Rectangle size between 5% and 20% of image dimensions
                min_size = int(256 * 0.05)
                max_size = int(256 * 0.2)
                
                # Random rectangle dimensions
                rect_h = np.random.randint(min_size, max_size)
                rect_w = np.random.randint(min_size, max_size)
                
                # Random position
                y = np.random.randint(0, 256 - rect_h)
                x = np.random.randint(0, 256 - rect_w)
                
                # Add rectangle to mask
                mask_tensor[0, 0, y:y+rect_h, x:x+rect_w] = 1.0
        
        # Generate defect image
        with torch.no_grad():
            # 使用修改后的推理函数，传递缺陷类型
            defect_img = fixed_inference(
                model=model,
                clean_image=img_tensor,
                mask=mask_tensor,
                object_class=args.object_class,
                defect_type=defect_type,  # 传递缺陷类型
                num_samples=args.num_samples,
                steps=args.steps,
                guidance_scale=args.guidance_scale
            )
        
        # Save generated image
        image_name = os.path.basename(args.image_path)
        output_path = os.path.join(args.output_dir, f"defect_{image_name}")
        save_image((defect_img + 1) / 2, output_path)
        
        # Also save the mask and original for comparison
        save_image(mask_tensor, os.path.join(args.output_dir, f"mask_{image_name}"))
        save_image((img_tensor + 1) / 2, os.path.join(args.output_dir, f"original_{image_name}"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with DefectFill model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Directory to save generated images")
    parser.add_argument("--object_class", type=str, required=True, help="Object class")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA adaptation")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha for LoRA scaling")
    # 添加缺陷类型参数
    parser.add_argument("--defect_type", type=str, help="Type of defect (e.g., 'broken_large')")
    
    # Image input options (either directory or single image)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_dir", type=str, help="Directory containing images")
    input_group.add_argument("--image_path", type=str, help="Path to single image")
    
    # Mask input options
    parser.add_argument("--mask_dir", type=str, help="Directory containing masks (if using image_dir)")
    parser.add_argument("--mask_path", type=str, help="Path to single mask (if using image_path)")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate for LPIPS selection")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    
    args = parser.parse_args()
    inference(args)