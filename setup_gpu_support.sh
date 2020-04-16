# PART A: on linux, reinstall torch torchvision (for cuda support):
pip uninstall torch torchvision
pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# PART B: install apex (note: python3-dev needs to be installed, sudo apt-get install python3-dev)
export CUDA_HOME=/usr/local/cuda-9.2
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
