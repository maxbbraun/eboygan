FROM tensorflow/tensorflow:latest-gpu-py3

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y git python3.6
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN easy_install3 pip
RUN pip3 install numpy requests tensorflow-gpu absl-py Pillow

RUN git clone https://github.com/NVlabs/stylegan.git
WORKDIR /stylegan

ADD eboy_data.py .
ADD eboy_data.json .
ADD eboy_train.py .
ADD eboy_run.sh .

ENTRYPOINT ["eboy_run.sh"]
