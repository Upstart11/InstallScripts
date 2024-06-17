#!/bin/bash

sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install software-properties-common
sudo apt-get --assume-yes install jq
sudo apt-get --assume-yes install build-essential
sudo apt-get --assume-yes install linux-headers-$(uname -r)

#install and activate miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh
chmod +x Miniconda3-py310_23.11.0-2-Linux-x86_64.sh
./Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
source .bashrc

#install Python and pip
sudo apt-get --assume-yes install python3
sudo apt-get --assume-yes install python3-pip

#install esential packages
sudo apt-get --assume-yes install tmux
sudo apt-get --assume-yes install nvtop
sudo apt-get --assume-yes install btop
sudo apt-get --assume-yes install ranger
sudo apt-get --assume-yes install micro

git clone https://github.com/Upstart11/Utils.git

chmod +x Utils/AutoClrC.sh
chmod +x Utils/ClrC.sh
chmod +x Utils/CheckIfCUIIsRunning.sh

chmod +x InstallScripts/ForVM/CopyConfigs.sh
./InstallScripts/ForVM/CopyConfigs.sh

chmod +x InstallScripts/ForVM/AddCronJobs.sh
./InstallScripts/ForVM/AddCronJobs.sh

source .bashrc


#confirm there is a Nvidia GPU
if lspci | grep -i -q nvidia; then
    echo "NVIDIA GPU found. Proceeding with installation."
    nvidia-smi
else
    echo "Error: No NVIDIA GPU found. Aborting installation."
    exit 1
fi

# install nvidia drivers and CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

#install torchvision and audio
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

#get user name and user home
USER_NAME=$(whoami)
USER_HOME=$(eval echo ~$USER_NAME)

#setting directories and paths
echo "Current directory: $(pwd)"
script_dir=$(dirname "$(realpath "$0")")
echo "script_dir : $script_dir"
parent_dir=$(dirname "$script_dir")
echo "parent_dir Directory: $parent_dir"
parentparent_dir=$(dirname "$parent_dir")
echo "parentparent_dir Directory: $parentparent_dir"
comfy_ui_dir="$parentparent_dir/ComfyUI"
echo "ComfyUI Directory: $comfy_ui_dir"

PYTHON_PATH="${USER_HOME}/miniconda3/bin/python"
BASHRC_PATH="${USER_HOME}/.bashrc"
COMFYUI_RUNNER_PATH="${comfy_ui_dir}/main.py"
COMFYUI_PATH=$comfy_ui_dir

#install CofyUI
git clone https://github.com/comfyanonymous/ComfyUI
pip install -r ComfyUI/requirements.txt

#install extensions
cd ComfyUI
#COMFY MANAGER
git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager
pip install -r custom_nodes/ComfyUI-Manager/requirements.txt

#install Checkpoints
mkdir -p ./models/clip/
mkdir -p ./models/clip_vision/
mkdir -p ./models/ipadapter/
mkdir -p ./models/upscale_models/
mkdir -p ./models/sams/
mkdir -p ./models/grounding_dino/
mkdir -p ./models/inpaint/

# dreamshaper 8
wget https://huggingface.co/autismanon/modeldump/resolve/main/dreamshaper_8.safetensors -P ./models/checkpoints/
