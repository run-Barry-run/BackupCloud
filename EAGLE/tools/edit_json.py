import json
from tqdm import tqdm
anno_json = json.load(open('dataset/ScanQA/ScanQA_v1.0_val.json'))
old_json = json.load(open('output/eval/onellm/eval_scanqa_.json'))

new_json_path = 'output/eval/onellm/eval_scanqa.json'

json_len = len(anno_json)

assert json_len == len(old_json)

new_json = []

for i in tqdm(range(json_len)):
    anno_dict = anno_json[i]
    old_dict = old_json[i]
    new_dict = {
        "question": anno_dict["question"],
        "answer": anno_dict["answers"],
        "prediction": old_dict["answer"],
        "question_id": anno_dict["question_id"]
    }
    new_json.append(new_dict)

with open(new_json_path, 'w') as f:
    json.dump(new_json, f)