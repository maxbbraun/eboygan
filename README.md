# eBoyGAN

It's [StyleGAN](https://github.com/NVlabs/stylegan2) + [eBoy](http://hello.eboy.com). Learn more on [Twitter](https://twitter.com/maxbraun/status/1137510117631389698).

![](eboygan_10x_7bb82c5d500be1f5366e91a2bf9d2e13_1.0psi_23i_20fps_256px.gif)
![](eboygan_10x_39df54ecab310cc97af7d6d686d7d163_1.0psi_23i_20fps_256px.gif)
![](eboygan_10x_960189ca27a3c66cd8cd02c56370d4e5_1.0psi_23i_20fps_256px.gif)

## Inference

Generate videos or GIFs right from the browser using [Colab](https://colab.research.google.com/drive/1IXI9cBgqS1_4A9Quhhve3B7uPMshauKX#forceEdit=true&offline=true&sandboxMode=true).

## Training

#### Create the VM

```
ZONE=us-central1-a
INSTANCE=eboygan-vm
gcloud compute instances create $INSTANCE \
    --zone=$ZONE \
    --image-family=tf-1-15-cu100 \
    --image-project=deeplearning-platform-release \
    --machine-type=n1-highmem-16 \
    --boot-disk-size=1TB \
    --accelerator="type=nvidia-tesla-v100,count=8" \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE \
    --scopes=default,storage-rw
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
git clone https://github.com/NVlabs/stylegan2.git
cd stylegan2
cp ../eboy_generate.py .
cp ../eboy_data.json .
IMAGES_DIR="$(pwd)/eboy-images"
DATASETS_DIR="$(pwd)/datasets"
DATASET_DIR="${DATASETS_DIR}/eboy"
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
nohup python run_training.py --num-gpus=8 --data-dir=$DATASETS_DIR \
    --result-dir=$RESULTS_DIR --dataset=eboy --config=config-f \
    --mirror-augment=true --total-kimg=25000 --metrics=fid50k &
```

#### Save the model checkpoints

```
RESULTS_BUCKET="gs://eboygan-results"
LOCATION="US-CENTRAL1"
gsutil mb -l $LOCATION $RESULTS_BUCKET
gsutil -m rsync -r $RESULTS_DIR $RESULTS_BUCKET
```

#### Stop the VM

```
gcloud compute instances stop $INSTANCE
```
