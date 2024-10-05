import json
from PIL import Image
from tqdm import tqdm
import os

def get_image_size(file_path):
    try:
        with Image.open(file_path) as img:
            return img.size  # returns (width, height)
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        return None

def convert_megadetector_to_coco(megadetector_json):
    coco_json = {
        "info": {
            "description": "Converted from MegaDetector output to COCO format",
            "year": 2024,
            "version": "1.0",
            "contributor": "Atharv",
            "date_created": "2024-06-13"
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "animal"},
            {"id": 2, "name": "person"},
            {"id": 3, "name": "vehicle"}
        ],
        "images": [],
        "annotations": []
    }
    
    image_id_counter = 0
    annotation_id_counter = 0
    image_id_map = {}  # To store mapping from file path to COCO image ID

    failed_dets = []
    err_img = []
    no_dets = []
    
    # Process each image in MegaDetector output
    for image_data in tqdm(megadetector_json["images"]):
        if "failure" in image_data:
            failed_dets.append(image_data)
            continue  # Skip failed images
        
        file_name = image_data["file"]
        detections = image_data.get("detections", [])
        file_path = os.path.join("../datasets/WII/wii.coco/images", file_name)
        
        # Get image dimensions
        img_size = get_image_size(file_path)
        if img_size is None:
            err_img.append(file_name)
            continue  # Skip if unable to read image

        width, height = img_size
        
        # Create COCO image entry
        image_info = {
            "id": image_id_counter,
            "file_name": file_name,
            "width": width,
            "height": height,
        }
        
        coco_json["images"].append(image_info)
        image_id_map[file_name] = image_id_counter
        image_id_counter += 1
        
        # Process detections
        # sort detections by score and keep only the first one
        # only keep those detections that have category 1)
        detections = [d for d in detections if int(d["category"]) == 1]
        if (len(detections) == 0):
            no_dets.append(file_name)
            continue
        detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
        detection = detections[0]

        category_id = int(detection["category"])
        bbox = detection["bbox"]
        conf = detection["conf"]

        # scale the bbox with image size
        bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        
        # COCO annotation format
        annotation = {
            "id": annotation_id_counter,
            "image_id": image_info["id"],
            "category_id": category_id,
            "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],  # COCO bbox format [x, y, width, height]
            "area": bbox[2] * bbox[3],  # Assuming bbox is in [x, y, w, h] format
            "iscrowd": 0,
            "segmentation": [],
            "attributes": {},
            "score": conf
        }
        
        coco_json["annotations"].append(annotation)
        annotation_id_counter += 1

    print(f"Failed detections: {len(failed_dets)}")
    print(f"Error reading images: {len(err_img)}")
    print(f"No detections: {len(no_dets)}")

    import pdb; pdb.set_trace()
    
    return coco_json

# Example usage:
if __name__ == "__main__":
    # Load MegaDetector JSON from file or directly from string
    with open('megadetector_output.json', 'r') as f:
        megadetector_json = json.load(f)
        print(f"Loaded MegaDetector output with {len(megadetector_json['images'])} images")
    
    # Convert to COCO format
    coco_json = convert_megadetector_to_coco(megadetector_json)
    
    # Save COCO JSON to file or use as needed
    with open('coco_annotations.json', 'w') as f:
        json.dump(coco_json, f, indent=4)
        print(f"Saved COCO annotations with {len(coco_json['images'])} images")
