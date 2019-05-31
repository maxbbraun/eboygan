#!/bin/bash

IMAGES_DIR=eboy-images
DATASET_DIR=eboy-dataset

# Download and crop the images.
python eboy_data.py --images_dir=$IMAGES_DIR

# Generate the dataset.
python dataset_tool.py create_from_images datasets/$DATASET_DIR $IMAGES_DIR

# Start training.
python eboy_train.py --dataset_dir=$DATASET_DIR
