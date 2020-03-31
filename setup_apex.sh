# needs to be installed: sudo apt-get install python3-dev
export CUDA_HOME=/usr/local/cuda-9.2
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
