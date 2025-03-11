import sys
sys.path.append('.')
from eagle.model.multimodal_encoder.languagebind.video.processing_video import load_and_transform_video

from eval.utils import replace_dir

from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

from tqdm import tqdm
import pyarrow as pa
import pyarrow.ipc as ipc
import os
import numpy as np

# pretrained = 'checkpoints/final_result1/video_finetune_1epoch'
# arrow_path = '.cache/huggingface/datasets/lmms-lab___you_cook2/default/0.0.0/d164962a70c383a57726388c9851e1486e9a9db7/you_cook2-val.arrow'
# video_dir = '.cache/huggingface/YouCookIIVideos/YouCookIIVideos/val'
# new_dir = 'dataset/eval/youcook2/videos/val'
video_dir = '.cache/huggingface/activitynetqa/all_test'
new_dir = 'dataset/eval/activitynetqa/videos'

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


transform_torch = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(8),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ShortSideScale(size=224),
            CenterCropVideo(224),
            RandomHorizontalFlipVideo(p=0.5),
        ]
    ),
)

transform_opencv = Compose(
    [
        # UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ShortSideScale(size=224),
        CenterCropVideo(224),
        RandomHorizontalFlipVideo(p=0.5),
    ]
)
transform_decord = Compose(
    [
        # UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ShortSideScale(size=224),
        CenterCropVideo(224),
        RandomHorizontalFlipVideo(p=0.5),
    ]
)
file_list = os.listdir(video_dir)
# error_indexs = [
#     114, 125, 374, 863, 3572, 4844
# ]
# error_list = [file_list[i] for i in error_indexs]
# print(error_list)
for i, old_file in tqdm(enumerate(file_list)):
    # if i in error_indexs:
    #     continue
    old_path = os.path.join(video_dir, old_file)
    new_path = os.path.join(new_dir, old_file)
    new_path = os.path.splitext(new_path)[0] + '.npy'
    if not os.path.exists(new_path):
        try:
            # feature = load_and_transform_video(
            #     video_path=old_path,
            #     transform=transform_decord,
            #     video_decode_backend='decord'
            # )
            feature = load_and_transform_video(
                video_path=old_path,
                transform=transform_torch,
                video_decode_backend='pytorchvideo'
            )['video']
            np.save(new_path, feature)
        except Exception as e:
            print('error at', i, ' ', old_file)
            print(e)
print('Pytorchvideo trans done')
# print('Decord test done')

# Read the .arrow file
# with pa.memory_map(arrow_path, 'r') as source:
#     # reader = ipc.RecordBatchFileReader(source)
#     reader = ipc.RecordBatchStreamReader(source)
#     # for batch in tqdm(reader):
#     #     for row in tqdm(batch.to_pylist()):
#     #         video_path = os.path.join(video_dir, row['video_path'])
#     #         try:
#     #             feature = load_and_transform_video(
#     #                 video_path=video_path,
#     #                 transform=transform_decord,
#     #                 video_decode_backend='decord'
#     #             )
#     #         except Exception as e:
#     #             print(e)
#     # print('Decord test done')

#     # for batch in tqdm(reader):
#     #     for row in tqdm(batch.to_pylist()):
#     #         video_path = os.path.join(video_dir, row['video_path'])
#     #         try:
#     #             feature = load_and_transform_video(
#     #                 video_path=video_path,
#     #                 transform=transform_opencv,
#     #                 video_decode_backend='opencv'
#     #             )
#     #         except Exception as e:
#     #             print(e)
#     # print('Opencv test done')

#     for batch in tqdm(reader):
#         for row in tqdm(batch.to_pylist()):
#             video_path = os.path.join(video_dir, row['video_path'])
#             new_path = os.path.join(new_dir, row['video_path']).replace('mp4', 'npy')
#             try:
#                 feature = load_and_transform_video(
#                     video_path=video_path,
#                     transform=transform_torch,
#                     video_decode_backend='pytorchvideo'
#                 )['video']
#                 np.save(new_path, feature)
#             except Exception as e:
#                 print(e)
#     print('Pytorchvideo test done')
