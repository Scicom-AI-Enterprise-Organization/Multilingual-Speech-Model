pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install neucodec==0.0.4 --no-deps
pip3 install torchao==0.12.0 torchtune==0.3.1 vector-quantize-pytorch==1.17.8 local_attention==1.11.1
python3 - <<'EOF'
from neucodec import NeuCodec

codec = NeuCodec.from_pretrained("neuphonic/neucodec")
codec = codec.eval().to('cuda')
EOF