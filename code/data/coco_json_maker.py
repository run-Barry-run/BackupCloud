"""
Turn annotation struct into 
{
    "annotations": [
        {
            "image_id": ,
            "caption": ,
            "id": 
        },
        ...
    ]
}
where "id" is unique, while "image_id" can be duplicated.
"""
import json
import os

datasets = [
    "VALOR-32K"
]

for dataset in datasets:
    if dataset == "VALOR-32K":
        ann_list = []
        json_data = None
        # TODO json_data = open("VALOR-32K-annotations.zip/desc_test.json")
        origin_json = json.load(json_data)
        for i, json_dict in enumerate(origin_json):
            ann_dict = {
                "image_id": json_dict["video_id"],
                "caption": json_dict["desc"],
                "id": i
            }
            ann_list.append(ann_dict)

        json_output = {
            "annotations": ann_list
        }
        # TODO fill the path
        gt_path = "eval_valor_gt.json"
        with open(gt_path, 'w') as f:
            json.dump(json_output, f)

