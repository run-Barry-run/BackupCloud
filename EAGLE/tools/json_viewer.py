import json

json_path = '/home1/hxl/disk/EAGLE/dataset/Eagle-1.8M/eagle-1509586.json'
old_json_path = '/home1/hxl/disk/EAGLE/dataset/Eagle-1.8M/eagle-1-sft-1_8M.json'

data = json.load(open(json_path))
print(data[0])
