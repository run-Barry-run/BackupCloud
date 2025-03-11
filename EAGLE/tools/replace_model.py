# 从 Safetensor 文件中加载参数
safetensor_path = "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors"
tensors = {}
with safe_open(safetensor_path, framework="pt", device='cpu') as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

# 替换模型参数
for name, param in model.named_parameters():
    if "vision_tower" not in name and "mm_projector" not in name:
        if name in tensors.keys():
            if tensors[name].shape == param.data.shape:
                original_requires_grad = copy.deepcopy(param.requires_grad)  # 保存原始的requires_grad状态
                original_device = copy.deepcopy(param.device)
                original_dtype = copy.deepcopy(param.dtype)
                param.data.copy_(tensors[name])
                param.requires_grad = original_requires_grad  # 恢复requires_grad状态
                param = param.to(original_device, dtype=original_dtype)
            else:
                print(f"Shape mismatch for {name}: {tensors[name].shape} != {param.data.shape}")
        else:
            print(f"Key {name} not found in safetensor")