# eBoyGAN

[StyleGAN](https://github.com/NVlabs/stylegan) + [eBoy](http://hello.eboy.com)

#### Create the VM
```
ZONE=us-central1-a
gcloud compute instances create eboygan-vm \
    --zone=$ZONE \
    --image-family=tf-latest-gpu \
    --image-project=deeplearning-platform-release \
    --accelerator="type=nvidia-tesla-v100,count=1" \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE
```

#### Connect to the VM (and forward [TensorBoard](http://localhost:6006) port)
```
gcloud compute ssh eboygan-vm --zone=$ZONE -- -NfL 6006:localhost:6006
gcloud compute ssh eboygan-vm --zone=$ZONE
```

#### Install dependencies
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y git python3.6
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
sudo easy_install3 pip
sudo pip3 install numpy requests tensorflow-gpu absl-py Pillow
```

#### Check out the code
```
git clone git@github.com:maxbbraun/eboygan.git
cd eboygan
git clone https://github.com/NVlabs/stylegan.git
cd stylegan
cp ../eboy_generate.py .
cp ../eboy_data.json .
cp ../eboy_train.py train.py
IMAGES_DIR="$(pwd)/eboy-images"
DATASET_DIR="$(pwd)/datasets/eboy"
RESULTS_DIR="$(pwd)/results"
```

#### Generate the training images
```
python generate.py --images_dir=$IMAGES_DIR
python dataset_tool.py create_from_images $DATASET_DIR $IMAGES_DIR
```

#### Start training
```
tensorboard --logdir=$RESULTS_DIR &
python train.py
```
