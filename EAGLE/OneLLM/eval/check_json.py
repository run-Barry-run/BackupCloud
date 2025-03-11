import json
import os

json_path = '/home1/hxl/Documents/ServerBackup/OneLLM/eval/results/eval_chartqa.json'

json_data = json.load(open(json_path))

accuracy = 0
strict = 0

for i in json_data:
    pred = i['answer']
    truth = i['gt_answer']
    if (pred.lower() in truth.lower()) or (truth.lower() in pred.lower()):
        accuracy += 1
    if pred.lower() == truth.lower():
        strict += 1

# print('Accuracy: ', accuracy / len(json_data))
# print('Strict Accuracy: ', strict / len(json_data))
print(accuracy / len(json_data))
print(strict / len(json_data))
