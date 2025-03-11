import sys
sys.path.append('./')
import os
import json
from tqdm import tqdm
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from fairscale.nn.model_parallel import initialize as fs_init
from util.misc import default_tensor_type
from util.misc import setup_for_distributed
import numpy as np
from model.meta import MetaModel
from data.conversation_lib import conv_templates
import torchvision.transforms as transforms


T_resized_center_crop = transforms.Compose([
    transforms.Resize(
        224, interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])



class CaptionDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = json.load(
            open('datasets/Eval/image/DocVQA/val_v1.0_withQT.json')
        )['data']
        self.img_dir = 'datasets/Eval/image/ChartQA/image'
        self.add_prompt = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        img_path = os.path.join(self.img_dir, data['image'])
        image = Image.open(img_path).convert('RGB')
        image = T_resized_center_crop(image)

        if self.add_prompt:
            question = data['question'] + '\nAnswer the question using a single word or phrase.'
        else:
            question = data['question']
        answer = data['answers']
        return image, question, answer


if __name__ == "__main__":
    pretrained_path = "weights/consolidated.00-of-01.pth"
    answer_path = "eval/results/eval_chartqa.json"
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)    
    
    mp.set_start_method("spawn")
    dist.init_process_group(
        backend="nccl", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:23563")
    fs_init.initialize_model_parallel(1)
    torch.cuda.set_device(3)
    torch.manual_seed(1)
    np.random.seed(1)
    # set the print behavior.
    setup_for_distributed(True)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }['fp16']
    with default_tensor_type(dtype=target_dtype, device="cuda"):
        model = MetaModel("onellm", "config/llama2/7B.json", None, "config/llama2/tokenizer.model")
       
    print("Loading pretrained weights ...")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print("load result:\n", msg)
    model.half().cuda()
    model.eval()
    print(f"Model = {str(model)}")

    def multi_modal_generate(images, inps):
        images = images.cuda().to(target_dtype)

        prompts = []
        for inp in inps:
            conv = conv_templates["v1"].copy()        
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())

        with torch.cuda.amp.autocast(dtype=target_dtype):
            responses = model.generate(prompts, images, 128, temperature=0.1, top_p=0.75, modal=['image'])
            outputs = []
            for response, prompt in zip(responses, prompts):
                response = response[len(prompt):].split('###')[0]
                response = response.strip()
                outputs.append(response)
        return outputs

    result = {}
    print("Starting...")
    dataset = CaptionDataset()
    dataloader = DataLoader(dataset, batch_size=28, shuffle=False, drop_last=False)
    print("Use additional prompt: ", dataset.add_prompt)
    predictions = []
    correct = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, questions, answers = data
            preds = multi_modal_generate(images, questions)
            for question, pred, answer in zip(questions, preds, answers):
                predictions.append({'answer': pred, 'gt_answer': answer})
                pred = pred.strip().lower()
                answer = answer.strip().lower()
                if (pred in answer) or (answer in pred):
                    correct += 1
    
    acc = float(correct) / len(dataset)
    print('Accuracy:', acc) 

    with open(answer_path, 'w') as f:
        json.dump(predictions, f)