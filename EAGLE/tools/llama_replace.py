import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model name
model_name = "model/LLM/Llama-2-7b-hf"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# print(model.state_dict().keys())
# model.save_pretrained("OneLLM/weights/LLM")

replace_dict = torch.load('OneLLM/weights/llm/pytorch_model.bin')
# new_dict = {}

# for k, v in replace_dict.items():
#     if 'attention' in k:
#         k = k.replace('attention', 'self_attn')
#         if 'wq' in k:
#             k = k.replace('wq', 'q_proj')
#         if 'wk' in k:
#             k = k.replace('wk', 'k_proj')
#         if 'wv' in k:
#             k = k.replace('wv', 'v_proj')
#         if 'wo' in k:
#             k = k.replace('wo', 'o_proj')
#     if 'feed_forward' in k:
#         k = k.replace('feed_forward', 'mlp')
#         if 'w1' in k:
#             k = k.replace('w1', 'gate_proj')
#         if 'w2' in k:
#             k = k.replace('w2', 'down_proj')
#         if 'w3' in k:
#             k = k.replace('w3', 'up_proj')
#     if 'self_attn_norm' in k:
#         k = k.replace('self_attn_norm', 'input_layernorm')
#     if 'ffn_norm' in k:
#         k = k.replace('ffn_norm', 'post_attention_layernorm')
#     new_dict[k] = v
#     print(k)

# torch.save(new_dict, 'OneLLM/weights/llm/pytorch_model_.bin')


# msg = model.model.layers.load_state_dict(replace_dict, strict=False)
# print("load result:\n", msg)

other_dict = torch.load('OneLLM/weights/llm/others.pth')

# msg = model.load_state_dict(other_dict, strict=False)
# print("\nother load result:\n", msg)

new_dict = {}
for k, v in other_dict.items():
    new_dict[k] = v
    print(k)

for k, v in replace_dict.items():
    k = 'model.layers.' + k
    new_dict[k] = v
    print(k)

# model.save_pretrained("OneLLM/weights/LLM")

torch.save(new_dict, 'model/LLM/Llama-2-7b-hf/pytorch_model_.bin')