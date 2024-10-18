conda create -n cumo_cuda12.1 python=3.10
conda activate cumo_cuda12.1
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 xformers -c xformers -c pytorch -c nvidia
conda install transformers scikit-learn requests tensorboard imageio tqdm scikit-image pyyaml pandas tensorboardx matplotlib
cd apex/
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..
cd flash-attention/
pip install ninja
rm -r build
python setup.py install
cd ..
pip install deepspeed==0.15.1 wandb
pip install -e -v ./