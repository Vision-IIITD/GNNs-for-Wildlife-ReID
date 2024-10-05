from megadetector.detection.run_detector_batch import \
  load_and_run_detector_batch, write_results_to_file
from megadetector.utils import path_utils
import os

# Pick a folder to run MD on recursively, and an output file
image_folder = os.path.expanduser('../datasets/WII/wii.coco/images/')
output_file = os.path.expanduser('megadetector_output.json')

# Recursively find images
image_file_names = path_utils.find_images(image_folder,recursive=False)

# This will automatically download MDv5a; you can also specify a filename.
results = load_and_run_detector_batch('MDV5A', image_file_names)

# Write results to a format that Timelapse and other downstream tools like.
write_results_to_file(results,
                      output_file,
                      relative_path_base=image_folder)
