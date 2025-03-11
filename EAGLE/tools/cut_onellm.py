import sys
sys.path.append('./')
import torch

# ckpt = torch.load('./weights/llm/llma_layers.pth')
# torch.save(ckpt, './weights/llm/pytorch_model.bin')

from model.meta import MetaModel
from util.misc import default_tensor_type


pretrained_path = "./weights/consolidated.00-of-01.pth"
with default_tensor_type(dtype=torch.float16, device="cuda"):
    model = MetaModel("onellm", "config/llama2/7B.json", None, "config/llama2/tokenizer.model")
    
print("Loading pretrained weights ...")
checkpoint = torch.load(pretrained_path, map_location='cpu')
msg = model.load_state_dict(checkpoint, strict=False)
print("load result:\n", msg)
model.half().cuda()
model.eval()
# print(f"Model = {str(model)}")

# save_checkpoint = {}
# for k, v in model.llma.tok_embeddings.state_dict().items():
#     save_checkpoint[f"model.embed_tokens.{k}"] = v
# for k, v in model.llma.output.state_dict().items():
#     save_checkpoint[f"lm_head.{k}"] = v
# for k, v in model.llma.norm.state_dict().items():
#     save_checkpoint[f"model.norm.{k}"] = v

# torch.save(save_checkpoint, "./weights/llm/others.pth")