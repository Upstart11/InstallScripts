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
sudo apt-get --assume-yes install nano

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

#COMFY MANAGER
git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager
pip install -r custom_nodes/ComfyUI-Manager/requirements.txt

#controlnet_aux
git clone https://github.com/Fannovel16/comfyui_controlnet_aux custom_nodes/comfyui_controlnet_aux
pip install -r custom_nodes/comfyui_controlnet_aux/requirements.txt

#IPADAPTER
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus custom_nodes/ComfyUI_IPAdapter_plus

# reactor
git clone https://github.com/Gourieff/comfyui-reactor-node custom_nodes/comfyui-reactor-node
python custom_nodes/comfyui-reactor-node/install.py
wget -P ~/models/facerestore_models https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
wget -P ~/models/facerestore_models https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

# WAS node nodes
git clone https://github.com/WASasquatch/was-node-suite-comfyui custom_nodes/was-node-suite-comfyui
pip install -r custom_nodes/was-node-suite-comfyui/requirements.txt

# WAS extra nodes
git clone https://github.com/WASasquatch/WAS_Extras custom_nodes/WAS_Extras
python custom_nodes/WAS_Extras/install.py

#image-resize-comfyui
git clone https://github.com/palant/image-resize-comfyui custom_nodes/image-resize-comfyui

# Impact-Pack
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack custom_nodes/ComfyUI-Impact-Pack
python custom_nodes/ComfyUI-Impact-Pack/install.py

#Face restore
mkdir -p ./models/facerestore_models/
git clone https://github.com/mav-rik/facerestore_cf custom_nodes/facerestore_cf
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth -P ./models/facerestore_models/
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth -P ./models/facerestore_models/

#install Checkpoints
mkdir -p ./models/clip/
mkdir -p ./models/clip_vision/
mkdir -p ./models/ipadapter/
mkdir -p ./models/upscale_models/

# turbovision xl
wget https://huggingface.co/akshitapps/TurboVisionXL/resolve/main/turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors -P ./models/checkpoints/

# dreamshaper 8
wget https://huggingface.co/autismanon/modeldump/resolve/main/dreamshaper_8.safetensors -P ./models/checkpoints/

# dreamshaper xl
wget https://civitai.com/api/download/models/251662 --content-disposition -P ./models/checkpoints/

#upscalers
wget https://civitai.com/api/download/models/357054 --content-disposition -P ./models/upscale_models/
wget https://civitai.com/api/download/models/156841 --content-disposition -P ./models/upscale_models/
wget https://civitai.com/api/download/models/125843 --content-disposition -P ./models/upscale_models/


# ipadapters
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors -P ./models/ipadapter/
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors -P ./models/ipadapter/
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors -P ./models/ipadapter/
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors -P ./models/ipadapter/
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors -P ./models/clip_vision/

# ControlNets
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_scribble_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_seg_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors -P ./models/controlnet/
wget https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11u_sd15_tile_fp16.safetensors -P ./models/controlnet/

## ControlNet SDXL
wget https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors -P ./models/controlnet/
wget https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors -P ./models/controlnet/
wget https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors -P ./models/controlnet/
wget https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors -P ./models/controlnet/
wget -O ./models/controlnet/depth-sdxl-1.0-diffusion_pytorch_model.bin https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-mid/resolve/main/diffusion_pytorch_model.bin

#vae
wget https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.ckpt -P ./models/vae/

 # clip vision
wget -O  ./models/clip_vision/SD15_CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors   https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors
