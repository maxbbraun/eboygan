# eBoyGAN

[StyleGAN](https://github.com/NVlabs/stylegan) + [eBoy](http://hello.eboy.com)

![](eboygan_f844e2620c0cf5f1e5a5748f54e5f8a2_50i_20fps_256px.gif)
![](eboygan_5d3bd2b4cc05cba5a69f0c96493dd255_50i_20fps_256px.gif)
![](eboygan_3a66d0c6d1b9a9f18e5cffc10cfe8c5b_50i_20fps_256px.gif)

## Inference

See [Colab](https://colab.research.google.com/drive/1IXI9cBgqS1_4A9Quhhve3B7uPMshauKX#forceEdit=true&offline=true&sandboxMode=true&scrollTo=t8OJLyhRgzpT)

## Training

#### Create the VM

```
ZONE=us-central1-a
INSTANCE=eboygan-vm
gcloud compute instances create $INSTANCE \
    --zone=$ZONE \
    --image-family=tf-latest-gpu \
    --image-project=deeplearning-platform-release \
    --machine-type=n1-highmem-16 \
    --boot-disk-size=1TB \
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
sudo update-alternatives --install /usr/bin/python python \
    /usr/local/bin/python3.6 2
sudo /usr/local/bin/pip3.6 install numpy scipy requests tensorflow-gpu absl-py \
    Pillow
```

#### Check out the code

```
git clone https://github.com/maxbbraun/eboygan.git
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
nohup tensorboard --logdir=$RESULTS_DIR > /dev/null 2>&1 &
nohup python train.py &
```

#### Stop the VM

```
gcloud compute instances stop $INSTANCE
```
