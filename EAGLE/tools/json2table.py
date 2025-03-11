import json
# json_path = './output/eval/final_result1__video_finetune_1epoch/20241209_143050_results.json'
json_path = './output/eval/weights__consolidated.00-of-01.pth/20241225_201903_results.json'
with open(json_path, 'r') as f:
    data = json.load(f)

result = data['results']
with open('./output/eval/weights__consolidated.00-of-01.pth/mvbench_results.json', 'w') as res:
    res_dict = {}
    for key in result.keys():
        if key == 'mvbench':
            continue
        res_dict[key] = result[key]['mvbench_accuracy,none']
    json.dump(res_dict, res)
