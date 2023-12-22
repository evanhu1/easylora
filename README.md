# lora-trainer
Automatically pulls dataset from GCS and train LoRA with LyCORIS / Kohya-SS

# Usage
```
python3.8 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
sudo .venv/bin/python3.8 train.py --lora_name=violet --install-dependencies --output=/home/np/stable-diffusion-webui/models/Lora
```
By default install_dependencies=False so feel free to omit that after the first run.
