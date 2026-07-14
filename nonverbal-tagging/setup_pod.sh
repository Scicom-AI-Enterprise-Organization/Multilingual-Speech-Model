#!/bin/bash
# One-time pod setup: deps + models + data. Run from /root.
set -e
cd /root

pip install -q --break-system-packages panns-inference librosa soundfile pandas pyarrow scipy \
  "faster-whisper>=1.0" "transformers>=4.44" "huggingface_hub[hf_transfer]" datasets tqdm

mkdir -p /root/models /root/data /root/out

# PANNs SED checkpoint (framewise). Try zenodo, fall back to HF mirror.
CKPT=/root/models/Cnn14_DecisionLevelMax.pth
if [ ! -f "$CKPT" ]; then
  wget -q --tries=5 --timeout=120 -O "$CKPT" \
    "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
fi
ls -la "$CKPT"

# warm CLAP + whisper caches
python3 - <<'EOF'
from transformers import ClapModel, ClapProcessor
ClapModel.from_pretrained("laion/clap-htsat-unfused")
ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
print("CLAP cached")
EOF
python3 - <<'EOF'
from faster_whisper import WhisperModel
WhisperModel("large-v3", device="cpu", compute_type="int8")
print("whisper cached")
EOF

df -h / | tail -1
echo "SETUP DONE"
