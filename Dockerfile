# FROM nvidia/cudagl:10.2-runtime-ubuntu18.04
FROM nvidia/cudagl:9.2-runtime-ubuntu18.04

SHELL ["/bin/bash", "-c"]

# Generic base installs
RUN apt-get update -q &&\
    apt-get install -y \
      git wget curl unzip python3-pip &&\
    apt-get autoclean -y &&\
    apt-get autoremove -y &&\
    rm -rf /var/lib/apt/lists/* &&\
    ln -s $(which python3) /usr/bin/python &&\
    ln -s $(which pip3) /usr/bin/pip

# Mujoco related installs
RUN apt-get update -q &&\
    apt-get install -y \
      libsm6 libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev &&\
    apt-get autoclean -y &&\
    apt-get autoremove -y &&\
    rm -rf /var/lib/apt/lists/* &&\
    curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf &&\
    chmod +x /usr/local/bin/patchelf &&\
    # ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so &&\
    mkdir -p /root/.mujoco &&\
    wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip &&\
    unzip mujoco.zip -d /root/.mujoco &&\
    mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 &&\
    rm mujoco.zip

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

RUN pip install 'numpy' 'tqdm' 'fire' 'ruamel.yaml'
RUN pip install 'torch'
RUN pip install 'tensorflow'
RUN pip install 'gym[all]' 'mujoco-py<2.1,>=2.0'

ADD ./mjkey.txt /root/.mujoco/

## NOTE: Uncomment this for pre-compilation of cpython objects
# RUN python -c "import mujoco_py"
