#!/bin/bash

IMAGES_DIR="$(pwd)/eboy-images"
DATASET_DIR="$(pwd)/datasets/eboy"

# Download and crop the images.
python eboy_data.py --images_dir=$IMAGES_DIR

# Generate the dataset.
python dataset_tool.py create_from_images $DATASET_DIR $IMAGES_DIR

# Start training.
python eboy_train.py --dataset_dir=$DATASET_DIR
