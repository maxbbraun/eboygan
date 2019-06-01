#!/bin/bash

IMAGES_DIR="$(pwd)/eboy-images"
DATASET_DIR="$(pwd)/datasets/eboy"
RESULTS_DIR="$(pwd)/results"

# Download and crop the images.
python eboy_data.py --images_dir=$IMAGES_DIR

# Generate the dataset.
python dataset_tool.py create_from_images $DATASET_DIR $IMAGES_DIR

# Launch TensorBoard in the background.
tensorboard --logdir=$RESULTS_DIR &

# Start training.
python train.py
