from safetensors.torch import safe_open
import torch

def compare_safetensors(file1, file2):
    # 使用safe_open加载safetensor文件
    tensors1 = {}
    tensors2 = {}
    with safe_open(file1, framework="pt") as model1, safe_open(file2, framework="pt") as model2:
        for k in model1.keys():
            tensors1[k] = model1.get_tensor(k)
        for k in model2.keys():
            tensors2[k] = model2.get_tensor(k)

    for k in tensors1.keys():
        if k in tensors2.keys():
            if tensors1[k].shape != tensors2[k].shape:
                print(f"参数不一样的层: {k}")
        else:
            print(f"参数不一样的层: {k}")

# 示例用法
compare_safetensors('checkpoints/Baseline/Video/Increamental/finetune-eagle-x1-llama3.2-1b-image_H_llama3.2/video_llama3.2-1b-finetune/model.safetensors', 
    'checkpoints/final_result1/video_finetune_1epoch/model.safetensors')