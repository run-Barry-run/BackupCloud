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
# import torch
# from eagle.model.multimodal_encoder.clip_encoder import LanguageBindVideo
# vision_tower = LanguageBindVideo.from_pretrained('./model/Vision_Encoder/LanguageBind/LanguageBind_Video_FT').vision_model
# input_ = torch.zeros((3, 8, 224, 224)).unsqueeze(0)
# print(input_.shape)
# with torch.no_grad():
#     output = vision_tower(pixel_values=input_)
#     print(output.shape)

import datasets

self_dataset = datasets.load_dataset(
    # path='/home1/hxl/disk/EAGLE/.cache/huggingface/datasets/lmms-lab___librispeech'
    # path='/home1/hxl/disk/EAGLE/.cache/huggingface/datasets/lmms-lab___clotho_aqa'
    # path='lmms-lab/ClothoAQA'
    path='/home1/hxl/disk/EAGLE/.cache/huggingface/datasets/lmms-lab___mme'
)
print(self_dataset['test'][0])
# # self_dataset = datasets.load_dataset(path='/home1/hxl/disk/EAGLE/.cache/huggingface/hub/datasets--hails--mmlu_no_train')
# # print(self_dataset)
# # audio = self_dataset['test'][0]['audio']
# audio = self_dataset['clotho_aqa_test_filtered'][0]['audio']
# array = audio['array']
# sampling_rate = audio['sampling_rate']
# print(array.shape)
# from lmms_eval.models.onellm import process_audio
# import torch
# image_tensor = process_audio(
#     array=torch.tensor(
#         array, 
#         # dtype=torch.double
#     ).unsqueeze(0), 
#     sampling_rate=torch.tensor(
#         sampling_rate, 
#         # dtype=torch.double
#     )
# ).unsqueeze(0) 
# print(image_tensor.shape)
# # print(len(array) / sampling_rate)
# print(audio)
# print(array)
# print(len(array))
# from eagle.model.multimodal_encoder.languagebind import LanguageBindAudioProcessor
# from eagle.model.multimodal_encoder.languagebind import LanguageBindAudio
# LanguageBindAudio_model = LanguageBindAudio.from_pretrained('/home1/hxl/disk/EAGLE/model/Vision_Encoder/LanguageBind/LanguageBind_Audio_FT')
# processor = LanguageBindAudioProcessor(LanguageBindAudio_model)
# import torchaudio
# # print(processor('/home1/hxl/disk/EAGLE/dataset/AudioSetCaps/example/_7Xe9vD3Hpg_4_10.mp3')['pixel_values'].shape)
# print(torchaudio.load('/home1/hxl/disk/EAGLE/dataset/AudioSetCaps/example/_7Xe9vD3Hpg_4_10.mp3'))

# import decord
# import numpy as np
# decord.bridge.set_bridge('torch')
# decord_vr = decord.VideoReader('2e121.mp4')
# duration = len(decord_vr)
# frame_id_list = np.linspace(0, duration-1, 8, dtype=int)
# video_data = decord_vr.get_batch(frame_id_list)
# video_data = video_data.permute(3, 0, 1, 2)
# print(video_data.shape)
# import torch
# a = torch.tensor([1, 1, 1])
# a = [a]
# print(torch.nn.utils.rnn.pad_sequence(a, batch_first=True))

# from lmms_eval.models.onellm import OneLLM
# import torch

# from eagle.model.builder import load_pretrained_model
# tokenizer, model, image_processor, max_length = load_pretrained_model('checkpoints/final_result1/3d_finetune_1epoch', model_base=None, model_name='eagle')
# print(model.state_dict())


# for i, img_data in tqdm(enumerate(data_loader, start=0)):
#     for key in img_data.keys():
#         print(key, ":", img_data[key].shape)
#     break

# import open_clip
# open_clip.create_model_and_transforms(
#             model_name='ViT-L-14',
#             pretrained='model/Vision_Encoder/CLIP/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin'
#         )
# from tqdm import tqdm
# from transformers import AutoTokenizer, LlamaForCausalLM, CLIPModel
# from torch.utils.data import DataLoader
# from Degrade.data.pretrain_dataset import make_supervised_data_module

# import torch
# llm_path = 'model/LLM/Llama-3.2-1B-Instruct'
# tokenizer=AutoTokenizer.from_pretrained(llm_path)
# print(tokenizer.pad_token_id)
# data_module = make_supervised_data_module(
#     tokenizer=tokenizer,
#     data_path='dataset/Eagle-1.8M/eagle-10.json',
#     image_folder='dataset/Eagle-1.8M'
# )
# llm = LlamaForCausalLM.from_pretrained(llm_path).to(torch.float32)
# data_loader = DataLoader(
#     data_module['train_dataset'],
#     batch_size=1,
#     num_workers=1,
#     drop_last=True,
#     prefetch_factor=16,
#     collate_fn=data_module['data_collator']
# )

# clip = CLIPModel.from_pretrained('model/Vision_Encoder/CLIP/CLIP-ViT-L-14-DataComp.XL-s13B-b90K').vision_model
# clip.to('cuda:0')
# llm.to('cuda:0')
# with torch.cuda.amp.autocast():
#     for i, img_data in tqdm(enumerate(data_loader, start=0)):
#         # for key in img_data.keys():
#         #     print(key, ":", img_data[key].shape)
#         pass
        # img_data['images'] = img_data['images'].to('cuda:0')
        # img_data['input_ids'] = img_data['input_ids'].to('cuda:0')
        # # print([tokenizer.decode(id_) for id_ in img_data['input_ids'][0] if id_ > 0])
        # # print(img_data['input_ids'] >= 0)
        # # print(img_data['input_ids'])
        # print(llm.model.embed_tokens.weight.shape)
        # x = clip(img_data['images'])
        # print(x)
        # h = llm.model.embed_tokens(img_data['input_ids']).to(torch.float32)
        # print(h)
        # break
        # x = clip.embeddings.patch_embedding(img_data['images'])
        # print(x)
        # break
# from tqdm import tqdm
# for i,a in tqdm(enumerate([1,1,1,1])):
#     print(i, a)

