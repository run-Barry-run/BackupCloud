from typing import Iterable, List
import json
import torch
import os
from io import BytesIO
import random
import copy
from torch.utils.data import Dataset
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from model.tokenizer import Tokenizer
import numpy as np
import warnings
import bisect

from .data_utils import make_audio_features, pc_norm, transform_pairimg_train, transform_img_train
from . import video_utils
from .imu_utils import get_imu_frames


DATASETS = dict(
    audio=dict(
        train=(
            './datasets/Eval/audio/clothov2/eval_clothocap_ann.json'
        ),
        test=None,
        max_words=96,
        data_path=(
            './datasets/Eval/audio/clothov2/evaluation'
        )
    )
)

JSON_STRUCT = dict(
    raw_data=dict(
        name='images',
        id_key='id',
        content_key='file_name'
    ),
    annotation=dict(
        name='annotations',
        data_id_key='image_id',
        id_key='id',
        content_key='caption'
    )
)

def get_dict_item(dict_list, id_key:str, id_value:int):

    return_list = []
    for item in dict_list:
        if item[id_key] == id_value:
            return_list.append(item)

    if len(return_list) == 1:
        return return_list[0]
    elif len(return_list) > 1:
        return return_list
    else:
        print(f'No such index: {id_value}')




class PretrainDataset(Dataset):
    def __init__(self, dataset='image', partition='train', epochs=1, tokenizer_path=None, petrel_conf=None):
        self.dataset = dataset

        self.petrel_conf = petrel_conf
        self.client = None
        self.partition = partition
        print('loading json...')

        input_file_path = DATASETS[dataset][partition]
        self.data_path = DATASETS[dataset]['data_path']

        print(input_file_path)
        
        data_json = json.load(open(input_file_path))

        self.datas = data_json[JSON_STRUCT['raw_data']['name']]
        self.annotations = data_json[JSON_STRUCT['annotation']['name']]

        self.max_words = 96
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        self.epochs = epochs

    def __len__(self):
        return int(len(self.datas) * self.epochs)
    
    def load_trans_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = transform_img_train(image)
        return image

    def load_audio(self, audio_path):
        fbank = make_audio_features(audio_path, mel_bins=128, aug=True)
        fbank = fbank.transpose(0, 1)[None] #[1, 128, 1024]
        return fbank
    
    def load_video(self, video_path):
        video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
        return video_feats[:, :, 0]

    def load_point(self, point_path):
        point_feat = np.load(point_path)
        # [8196, 6]
        point_feat = torch.tensor(point_feat)
        point_feat = pc_norm(point_feat)

        return point_feat
    
    def load_rgbx(self, image_path):
        replace_list = DATASETS[self.dataset]['replace_list']
        x_image_path = image_path.replace(replace_list[0], replace_list[1])
        image = Image.open(image_path).convert('RGB')
        x_image = Image.open(x_image_path).convert('RGB')
        x_image = x_image.resize(image.size[-2:])

        image, x_image = transform_pairimg_train([image, x_image])

        # [2, 3, H, W]
        image = torch.stack([image, x_image], dim=0)
        return image
    
    def load_fmri(self, fmri_path):
        data = np.load(fmri_path)
        data = data.mean(axis=0)
        data = torch.tensor(data[None])
        return data

    def load_imu(self, data_dict):
        uid = data_dict["video_uid"]
        w_s = data_dict["window_start"]
        w_e = data_dict["window_end"]

        imu_data = get_imu_frames(
            self.imu_path, uid,
            video_start_sec=w_s,
            video_end_sec=w_e,
        )
        if imu_data is None:
            raise ValueError
        return imu_data['signal']

    def __getitem__(self, index):
        index = index % len(self.datas)

        filename = get_dict_item(
            dict_list=self.datas,
            id_key='id',
            id_value=index,
        )[JSON_STRUCT['raw_data']['content_key']]

        data_path = os.path.join(self.data_path, filename)

        caption = get_dict_item(
            dict_list=self.annotations,
            id_key='image_id',
            id_value=index
        )

        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption[JSON_STRUCT['annotation']['content_key']])

        try:
            if self.dataset == 'image':
                data = self.load_trans_image_from_ceph(data_path)
            elif self.dataset == 'audio':
                data = self.load_audio(data_path)
            elif self.dataset == 'video':
                data = self.load_video_from_ceph(data_path)
            elif self.dataset == 'point':
                data = self.load_point(data_path)
            elif self.dataset in ['rgbn', 'rgbd']:
                data = self.load_rgbx(data_path)
            elif self.dataset == 'fmri':
                data = self.load_fmri(data_path)
            elif self.dataset == 'imu':
                data_dict = self.datas[index]
                data = self.load_imu(data_dict)
        except:
            print(data_path, 'Not Found')
            rand_idx = random.randint(0, len(self))
            return self.__getitem__(rand_idx)  
        
        caption_tokens = torch.tensor(self.tokenizer.encode(caption, bos=True, eos=True), dtype=torch.int64)
        input_data = caption_tokens

        padding = self.max_words - input_data.shape[0]
        if padding > 0:
            input_data = torch.cat((input_data, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            input_data = input_data[:self.max_words]
        labels = copy.deepcopy(input_data)

        if self.partition != 'train':
            return input_data, labels, data, data_path, self.dataset, caption
        
        return input_data, labels, data, data_path, self.dataset

    def __repr__(self):
        return f"<XTextPair:{self.dataset}"


class ConcatDataset(Dataset):
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        # for d in self.datasets:
            # assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    def get_indices(self, batch_size, world_size=1, rank_id=0):
        random.seed(0)
        real_batch_size = batch_size * world_size
        batch_train_indices = []
        num_batches = []
        for i in range(len(self.datasets)):
            # get train_indices
            start_idx = self.cumulative_sizes[i-1] if i>0 else 0
            end_idx = self.cumulative_sizes[i]
            train_indice = list(range(start_idx, end_idx))
            random.shuffle(train_indice)
            num_batch = int(len(self.datasets[i]) / real_batch_size)
            num_batches.append(num_batch)
            # get batch indices for each rank
            batch_train_indice = [
                train_indice[batch*real_batch_size:(batch+1)*real_batch_size][rank_id::world_size]
                for batch in range(num_batch)
            ]
            batch_train_indices.append(batch_train_indice)
        min_num_batch = min(num_batches)

        train_indices = []
        for batch in range(min_num_batch):
            for i in range(len(self.datasets)):
                train_indices.extend(batch_train_indices[i][batch])
        
        return train_indices