# from transformers import AutoConfig
# from eagle.model import *

# config = AutoConfig.from_pretrained("./model/Eagle-X5-7B")

# model = EagleLlamaForCausalLM.from_pretrained(
#     # "./model/Eagle-X5-7B",
#     # "./checkpoints/final_result/image/finetune-eagle-x1-llama3.2-1b-image_L",
#     "./checkpoints/final_result/video/video_finetune_1epoch",
#     low_cpu_mem_usage=True
# )
# print(model)
# from safetensors import safe_open

# tensors = {}
# with safe_open("./model/Eagle-X5-7B/model-00004-of-00004.safetensors", framework="pt", device=0) as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k).shape # loads the full tensor given a key
# print(tensors)
import torch
from eagle.model.multimodal_encoder.clip_encoder import LanguageBindVideo
vision_tower = LanguageBindVideo.from_pretrained('./model/Vision_Encoder/LanguageBind/LanguageBind_Video_FT').vision_model
input_ = torch.zeros((3, 8, 224, 224)).unsqueeze(0)
print(input_.shape)
with torch.no_grad():
    output = vision_tower(pixel_values=input_)
    print(output.shape)

# import datasets
# self_dataset = datasets.load_dataset(
#     # path='/home1/hxl/disk/EAGLE/.cache/huggingface/datasets/lmms-lab___librispeech'
#     path='/home1/hxl/disk/EAGLE/.cache/huggingface/datasets/lmms-lab___clotho_aqa'
# )
# audio = self_dataset['test'][0]['audio']
# array = audio['array']
# sampling_rate = audio['sampling_rate']
# # print(len(array) / sampling_rate)
# print(audio)
# print(len(array))
# from eagle.model.multimodal_encoder.languagebind import LanguageBindAudioProcessor
# from eagle.model.multimodal_encoder.languagebind import LanguageBindAudio
# LanguageBindAudio_model = LanguageBindAudio.from_pretrained('/home1/hxl/disk/EAGLE/model/Vision_Encoder/LanguageBind/LanguageBind_Audio_FT')
# processor = LanguageBindAudioProcessor(LanguageBindAudio_model)
# import torchaudio
# # print(processor('/home1/hxl/disk/EAGLE/dataset/AudioSetCaps/example/_7Xe9vD3Hpg_4_10.mp3')['pixel_values'].shape)
# print(torchaudio.load('/home1/hxl/disk/EAGLE/dataset/AudioSetCaps/example/_7Xe9vD3Hpg_4_10.mp3'))