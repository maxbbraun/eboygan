# eBoyGAN

[StyleGAN](https://github.com/NVlabs/stylegan) + [eBoy](http://hello.eboy.com)

```
docker build -t eboygan .
docker tag eboygan gcr.io/eboygan/eboygan
docker push gcr.io/eboygan/eboygan:latest
```

```
gcloud compute instances create-with-container eboygan-vm \
    --zone=us-central1-a \
    --container-image=gcr.io/eboygan/eboygan:latest \
    --maintenance-policy=TERMINATE \
    --accelerator="type=nvidia-tesla-v100,count=8" \
    --machine-type=n1-standard-8 \
    --boot-disk-size=120GB \
    --metadata="install-nvidia-driver=True"
```
