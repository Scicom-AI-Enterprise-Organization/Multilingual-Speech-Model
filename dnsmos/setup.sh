uv venv --python 3.12 --allow-existing
uv pip install onnxruntime librosa tqdm pandas huggingface_hub

wget https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx

source .venv/bin/activate
python dnsmos_hf.py --sample 10

