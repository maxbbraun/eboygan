# eBoyGAN

[StyleGAN](https://github.com/NVlabs/stylegan) + [eBoy](http://hello.eboy.com)

#### Create the VM
```
ZONE=us-central1-a
INSTANCE=eboygan-vm
gcloud compute instances create $INSTANCE \
    --zone=$ZONE \
    --image-family=tf-latest-gpu \
    --image-project=deeplearning-platform-release \
    --machine-type=n1-standard-8 \
    --accelerator="type=nvidia-tesla-v100,count=8" \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE
```

#### Connect to the VM (and forward [TensorBoard](http://localhost:6006) port)
```
gcloud compute ssh $INSTANCE --zone=$ZONE -- -NfL 6006:localhost:6006
gcloud compute ssh $INSTANCE --zone=$ZONE
```

#### Install dependencies
```
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev
wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
tar xvf Python-3.6.8.tgz
cd Python-3.6.8
./configure --enable-optimizations --with-ensurepip=install
make -j8
sudo make altinstall
cd ..
sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.6 2
sudo /usr/local/bin/pip3.6 install numpy requests tensorflow-gpu absl-py Pillow
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
python eboy_generate.py --images_dir=$IMAGES_DIR
python dataset_tool.py create_from_images $DATASET_DIR $IMAGES_DIR
```

#### Start training
```
tensorboard --logdir=$RESULTS_DIR &
python train.py
```

#### Stop the VM
```
gcloud compute instances stop $INSTANCE
```
