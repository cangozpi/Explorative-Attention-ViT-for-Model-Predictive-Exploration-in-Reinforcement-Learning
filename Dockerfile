FROM ubuntu:22.04

RUN apt update && apt upgrade -y

# WORKDIR /Docker_shared

# RUN mkdir Representation\ Learning\ for\ Model-Predictive\ Exploration\ in\ Reinforcement\ Learning

WORKDIR /Docker_shared/Representation\ Learning\ for\ Model-Predictive\ Exploration\ in\ Reinforcement\ Learning
COPY ./ ./

# install miniconda
# RUN apt install wget -y
# RUN mkdir -p ~/miniconda3 && \
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
#     bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
#     rm -rf ~/miniconda3/miniconda.sh
# RUN ~/miniconda3/bin/conda init bash && \
#     ~/miniconda3/bin/conda init zsh
# # enable conda command
# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# # create conda env 
# RUN ~/miniconda3/bin/conda --version
# RUN ~/miniconda3/bin/conda config --append channels conda-forge
# RUN conda create --name rnd python=3.8.16 --file requirements.txt
# install requirement to conda env
# RUN conda install -c conda-forge numpy==1.24.2
# RUN conda install -c conda-forge torch==2.0.1
# RUN conda install -c conda-forge torchvision==0.15.2
# RUN conda install -c conda-forge gym==0.26.2
# RUN conda install -c conda-forge gym-super-mario-bros==7.4.0
# RUN conda install -c conda-forge autorom==0.4.2
# RUN conda install -c conda-forge autorom-accept-rom-license==0.6.1
# RUN conda install -c conda-forge opencv-python==4.6.0.66
# RUN conda install -c conda-forge pillow==9.5.0
# RUN conda install -c conda-forge tensorboard==2.13.0
# RUN conda install -c conda-forge tensorboard-data-server==0.7.0
# RUN conda install -c conda-forge torch_tb_profiler

# install python 3.8.16
ARG DEBIAN_FRONTEND=noninteractive
RUN apt install software-properties-common -y # required to use add-apt-repository command
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.8 -y
# change python3 to refer to 3.8 instead of 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 20

# install pip23.0.1
RUN apt install python3-pip -y
RUN apt install python3-distutils -y
RUN apt install python3-apt -y
RUN apt install python3.8-distutils -y
RUN python3 -m pip install pip==23.0.1 # Upgrade to a specific version

# install python3 requirements:
RUN apt install ffmpeg libsm6 libxext6 -y
RUN pip3 install -r requirements.txt
# RUN pip3 install numpy==1.24.2
# RUN pip3 install torch==2.0.1
# RUN pip3 install torchvision==0.15.2
# RUN pip3 install gym==0.26.2
# RUN pip3 install gym-super-mario-bros==7.4.0
# RUN pip3 install autorom==0.4.2
# RUN pip3 install autorom-accept-rom-license==0.6.1
# RUN pip3 install opencv-python==4.6.0.66
# RUN pip3 install pillow==9.5.0
# RUN pip3 install tensorboard==2.13.0
# RUN pip3 install tensorboard-data-server==0.7.0
# RUN pip3 install torch_tb_profiler
# RUN pip3 install gym[atari,accept-rom-license]==0.26.2
# RUN pip3 install gym[classic_control]==0.26.2
# RUN pip3 install matplotlib

# #scalene + kornia
# RUN pip3 install kornia
# RUN pip3 install scalene


# activate conda env by default
# CMD['conda', 'activate', 'rnd']