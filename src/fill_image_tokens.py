import pandas as pd
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_image_data_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()

# Define a custom JSON encoder that handles NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
if __name__ == "__main__":
    args = parse_args()

    image_data_tokenized = pd.read_parquet(args.tokenized_image_data_path)
    print(image_data_tokenized.head())

    with open(args.metadata_path, "r") as f:
        text_data = [json.loads(line) for line in f]
        
    for instance in text_data:
        img_path = instance['img_path']
        if img_path == "":
            print("Skipping empty image path")
            continue
        else:
            image_tokens = image_data_tokenized[image_data_tokenized['img_path'] == img_path]['img_tokens'].values[0]
            # Convert NumPy array to standard Python list with native Python integers
            instance['image_tokens'] = [int(token) if isinstance(token, np.integer) else token for token in image_tokens]

    with open(args.output_path, "w") as f:
        for instance in text_data:
            f.write(json.dumps(instance, cls=NumpyEncoder) + "\n")
