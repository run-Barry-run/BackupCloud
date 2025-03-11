import torch
from PIL import Image
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True

import logging
import copy
from tqdm import tqdm
from datetime import timedelta

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

from eval.utils import replace_dir

from typing import List, Optional, Union, Tuple
import warnings

warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")
import torchvision.transforms as transforms
import torchaudio

# try:
from OneLLM.model.meta import MetaModel
from OneLLM.util.misc import default_tensor_type, setup_for_distributed
from OneLLM.data.conversation_lib import conv_templates
# except ImportError:
#     eval_logger.error("Please add a symbolic link pointing to the eagle folder of repo ")

from OneLLM.data import video_utils

def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]

def process_audio(array, sampling_rate, mel_bins=128, target_length=1024):
    if sampling_rate != 16000:
        trans = torchaudio.transforms.Resample(sampling_rate, 16000, dtype=torch.double)
        array = trans(array)

    array = array - array.mean()

    fbank = torchaudio.compliance.kaldi.fbank(
        array, htk_compat=True, sample_frequency=16000, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    fbank = fbank.transpose(0, 1)[None]
    return fbank

T_resized_center_crop = transforms.Compose([
    transforms.Resize(
        224, interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

def resize_image_with_aspect_ratio(img, min_size):
    """
    Resize an image while maintaining its aspect ratio.
    
    Parameters:
    - image_path: str, path to the input image.
    - min_size: int, the minimum size for the shortest side of the image.
    
    Returns:
    - resized_image: PIL.Image object, the resized image.
    """
    # Get the original dimensions of the image
    original_width, original_height = img.size

    # Determine the aspect ratio
    aspect_ratio = original_width / original_height
    
    # Calculate new dimensions based on the shortest side
    if original_width < original_height:
        new_width = min_size
        new_height = int(min_size / aspect_ratio)
    else:
        new_height = min_size
        new_width = int(min_size * aspect_ratio)
    
    # Resize the image while maintaining aspect ratio
    resized_image = img.resize((new_width, new_height), Image.LANCZOS)# Image.ANTIALIAS)
        
    return resized_image
    

@register_model("onellm")
class OneLLM(lmms):
    """
    EAGLE Model
    """

    def __init__(
        self,
        pretrained: str = "NVEagle/Eagle-X5-7B",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        revision=None,
        use_flash_attention_2=True,
        device_map="",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,
        modality='image',
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"


        self._device = torch.device(device)
        self.device_map = device_map

        self.target_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16
        }['fp16']
        with default_tensor_type(dtype=self.target_dtype, device="cuda"):
            self._model = MetaModel("onellm", "OneLLM/config/llama2/7B.json", None, "OneLLM/config/llama2/tokenizer.model")
            print("Loading pretrained weights ...")
        checkpoint = torch.load(pretrained, map_location='cpu')
        msg = self._model.load_state_dict(checkpoint, strict=False)
        print("load result:\n", msg)
        self._model.half().cuda()
        self._model.eval()
        print(f"Model = {str(self._model)}")

        self._tokenizer = self._model.tokenizer

        # BEGIN hxl add self modality
        self.modality = modality
        print('modality:', self.modality)
        # END
        # self._config = self._model.config
        self.model.eval()
        # self.model.tie_weights()
        # self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        # self.conv_template = conv_template
        # self.use_cache = use_cache
        # self.truncate_context = truncate_context

        eval_logger.info(f"Using single device: {self._device}")
        self.model.to(self._device)
        self._rank = 0
        self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, bos=True, eos=False)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        pass

    def generate_until(self, requests: List[Instance]) -> List[str]:
        # print(requests)
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        for chunk in chunks:
            # print(chunk)
            _, _, _, _, task_flag, _ = zip(*chunk)
            break
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        context_list = []
        doc_id_list = []
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

        
            # BEGIN hxl add try
            try:
                if visuals:
                    # print(visuals)
                    # print(len(visuals[0]['array']))
                    if self.modality == 'image':
                        image_tensor = Image.open(visuals[0]).convert('RGB')
                        # For image
                        image_tensor = T_resized_center_crop(image_tensor).unsqueeze(0)
                    elif self.modality == 'video':
                        # BEGIN hxl, replace npy in cache
                        # print(visuals)
                        if isinstance(visuals[0], str):
                            replace_visuals = replace_dir(visuals[0])
                            if replace_visuals != '':
                                image_tensor = torch.from_numpy(np.load(replace_visuals)).permute(1, 0, 2, 3).unsqueeze(0)
                            else:
                                print('ignore', visuals)
                                continue
                                image_tensor = process_images(visuals, self._image_processor, self._config)
                        # END
                        else:   
                            image_tensor = load_video(visuals[0]).unsqueeze(0)
                    elif self.modality == 'audio':
                        image_tensor = process_audio(
                            array=torch.tensor(visuals[0]['array']).unsqueeze(0), 
                            sampling_rate=torch.tensor(visuals[0]['sampling_rate'])
                        ).unsqueeze(0) 
                else:
                    image_tensor = None
            except Exception as e:
                print(e)
                print('Visuals error with', visuals)
                # assert False
                continue
            # END
            # prompts_input = contexts[0]

            question_input = []

            for visual, context in zip(visuals, contexts):

                conv = conv_templates["v1"].copy()
                conv.append_message(conv.roles[0], context)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # The above for loop has bugs. When there is no visuals, e.g. pure text,
            # there will be no for loop execute resulting in an empty question_input (because no visuals)
            # Scenario 1 won't even be execute
            if len(visuals) == 0:
                for context in contexts:
                    question = context
                    conv = conv_templates["v1"].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            # BEGIN hxl
            # Add try and add modal
            try:
                with torch.cuda.amp.autocast(dtype=self.target_dtype):
                    text_outputs = self.model.generate(
                        prompts=question_input,
                        images=image_tensor,
                        max_gen_len=128,
                        temperature=0.1,
                        top_p=0.75,
                        modal=[self.modality]
                    )
                    outputs = []
                    for text_output, prompt in zip(text_outputs, question_input):
                        text_output = text_output[len(prompt):].split('###')[0]
                        text_output = text_output.strip()
                        outputs.append(text_output)
            
            # print(image_tensor.shape)
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                outputs = [""]

            res.extend(outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), outputs)
            pbar.update(1)


        res = re_ords.get_original(res)

        pbar.close()
        # BEGIN hxl get result
        save_list = []
        for text, context, doc in zip(res, context_list, doc_id_list):
            save_list.append({
                "text_output": text,
                "context": context,
                "doc_id": doc
            })
        from datetime import datetime
        import os
        # 获取当前日期，格式化为字符串（例如：2023-10-05）
        current_date = datetime.now().strftime("%Y%m%d")
        output_dir = f'output/eval/{current_date}'
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/{task_flag}.json', 'w') as json_file:
            import json
            json.dump(save_list, json_file)
            print('result dumped')
        # END hxl
        return res
