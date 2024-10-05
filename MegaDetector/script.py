import json
import pandas as pd

input_json_path = "coco_annotations.json"
output_json_path = "../datasets/WII/wii.coco/instances_train2017.json"

with open(input_json_path) as f:
    megadetector_json = json.load(f)

csv_train = "/mnt/nas/WII-ReID/Train/wii_train.csv"
csv_train_5 = "/mnt/nas/WII-ReID/Train/wii_train_5.csv"
csv_test = "/mnt/nas/WII-ReID/Test/wii_test.csv"
csv_test_5 = "/mnt/nas/WII-ReID/Test/wii_test_5.csv"

df_train = pd.read_csv(csv_train)
# df_train_5 = pd.read_csv(csv_train_5)
df_test = pd.read_csv(csv_test)
# df_test_5 = pd.read_csv(csv_test_5)

# df_tigers = pd.read_csv("/mnt/nas/WII-ReID/wii_tigers.csv")
# df_tigers_5 = pd.read_csv("/mnt/nas/WII-ReID/wii_tigers_5.csv")
# # get the set of image names in df_tigers
# tigers = set(df_tigers["Image_Name"].values)
# tigers_5 = set(df_tigers_5["Image_Name"].values)

# For any image in the MegaDetector output, we need to find the corresponding image in the WII dataset
# This is done by matching the image file names

# Create a dictionary mapping image file names to image IDs in the WII dataset
ann_map = {}

for i, row in df_train.iterrows():
    ann_map[row["Image_Name"]] = (row["Entity"], "train")

# for i, row in df_train_5.iterrows():
#     image_id_map[row["Image_Name"]] = (row["Entity"], "train_5")

for i, row in df_test.iterrows():
    ann_map[row["Image_Name"]] = (row["Entity"], "test")

# for i, row in df_test_5.iterrows():
#     image_id_map[row["Image_Name"]] = (row["Entity"], "test_5")

image_id_map = {int(img["id"]): img for img in megadetector_json["images"]}

invalid_mappings = []

output_json = {**megadetector_json, "annotations": [], "images": []}
import pdb; pdb.set_trace()

for ann in megadetector_json["annotations"]:
    img_dict = image_id_map[int(ann["image_id"])]
    file_name = img_dict["file_name"]
    try:
        entity, split = ann_map[file_name]
    except KeyError:
        invalid_mappings.append(file_name)
        continue

    ann["identity"] = int(entity)
    ann["split"] = split

    output_json["annotations"].append(ann)
    output_json["images"].append(img_dict)

print(f"Invalid mappings: {len(invalid_mappings)}")

if len(invalid_mappings) > 0:
    import pdb; pdb.set_trace()

with open(output_json_path, "w") as f:
    json.dump(output_json, f, indent=2)

print("Done!")