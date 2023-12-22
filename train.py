import os
import re
import toml
from time import time
import argparse
from get_dataset import download_dataset

def get_config(project_name, do_install_dependencies):
    root_dir = os.path.abspath(os.getcwd())
    deps_dir = os.path.join(root_dir, "deps")
    kohya_trainer_repo_dir = os.path.join(deps_dir, "kohya-trainer")
    lycoris_repo_dir = os.path.join(deps_dir, "lycoris")
    log_folder    = os.path.join(root_dir, "_logs")
    models_folder = os.path.join(root_dir, "models")
    deps_dir = os.path.join(root_dir, "deps")
    loras_folder = os.path.join(root_dir, "loras")
    config_folder = os.path.join(loras_folder, project_name)
    images_folder = os.path.join(loras_folder, project_name, "dataset")
    output_folder = os.path.join(loras_folder, project_name, "output")
    config_file = os.path.join(config_folder, "training_config.toml")
    dataset_config_file = os.path.join(config_folder, "dataset_config.toml")
    accelerate_config_file = os.path.join(kohya_trainer_repo_dir, "accelerate_config/config.yaml")
    
    config = {
      "project_name": project_name,
      "GCS_BUCKET_NAME": "shortbreadresearch",
      "epochs": 10,
      "train_batch_size": 4,
      "num_repeats": 10,
      "model_url": "https://huggingface.co/a1079602570/animefull-final-pruned/resolve/main/novelailatest-pruned.ckpt?download=true", # https://huggingface.co/Lykon/AnyLoRA/resolve/main/AnyLoRA_noVae_fp16-pruned.ckpt
      "dependencies_installed": do_install_dependencies,
      "weighted_captions": False,
      "scheduler": "cosine_with_restarts",
      "optimizer": "prodigy", # AdamW8bit
      "unet_learning_rate": 1e-4,
      "te_learning_rate": 5e-5,
      "optimizer_args": [],
      "adjust_tags": False,
      "keep_tokens_weight": 1.0,
      "keep_tokens": 1,
      "BETTER_EPOCH_NAMES": True,
      "LOAD_TRUNCATED_IMAGES": True,
      "custom_model_is_based_on_sd2": False,
      "root_dir": root_dir,
      "deps_dir": deps_dir,
      "kohya_trainer_repo_dir": kohya_trainer_repo_dir,
      "lycoris_repo_dir": lycoris_repo_dir,
      "log_folder": log_folder,
      "models_folder": models_folder, 
      "loras_folder": loras_folder,
      "config_folder": config_folder,
      "images_folder": images_folder,
      "output_folder": output_folder,
      "config_file": config_file,
      "dataset_config_file": dataset_config_file,
      "accelerate_config_file": accelerate_config_file
    }
    if config["optimizer"] == "prodigy":
      config["optimizer_args"] = ["decouple=True","weight_decay=0.05","betas=0.9,0.999", "d_coef=2","use_bias_correction=True", "safeguard_warmup=False"]
      config["scheduler"] == "constant"
      config["unet_learning_rate"] = 1.0
      config["te_learning_rate"] = 1.0

    return config

def clone_repo(config):
  os.chdir(config["root_dir"])
  os.system("git clone https://github.com/kohya-ss/sd-scripts " + config["kohya_trainer_repo_dir"])
  os.system("git clone https://github.com/KohakuBlueleaf/LyCORIS " + config["lycoris_repo_dir"])
  os.chdir(config["kohya_trainer_repo_dir"])
  os.system("wget https://raw.githubusercontent.com/hollowstrawberry/kohya-colab/xformers-fix/requirements.txt -q -O requirements.txt")

def install_dependencies(config):
  clone_repo(config)
  os.system("apt -y update -qq")
  os.system("apt -y install aria2 -qq")
  # os.system("pip install --upgrade -r requirements.txt")
  # if XFORMERS:
  #   os.system("pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118")

  # patch kohya for minor stuff
  if config["LOAD_TRUNCATED_IMAGES"]:
    os.system("sed -i 's/from PIL import Image/from PIL import Image, ImageFile\\nImageFile.LOAD_TRUNCATED_IMAGES=True/g' library/train_util.py") # fix truncated jpegs error
  if config["BETTER_EPOCH_NAMES"]:
    os.system("sed -i 's/{:06d}/{:02d}/g' library/train_util.py") # make epoch names shorter
    os.system("sed -i 's/\".\" + args.save_model_as)/\"-{:02d}.\".format(num_train_epochs) + args.save_model_as)/g' train_network.py") # name of the last epoch will match the rest

  from accelerate.utils import write_basic_config
  if not os.path.exists(config["accelerate_config_file"]):
    write_basic_config(save_location=config["accelerate_config_file"])

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["BITSANDBYTES_NOWELCOME"] = "1"
  os.environ["SAFETENSORS_FAST_GPU"] = "1"

def validate_dataset(config):
  supported_types = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

  print("\n💿 Checking dataset...")
  if not config["project_name"].strip() or any(c in config["project_name"] for c in " .()\"'\\/"):
    print("💥 Error: Please choose a valid project name.")
    return

  reg = []
  folders = [config["images_folder"]]
  files = os.listdir(config["images_folder"])
  images_repeats = {config["images_folder"]: (len([f for f in files if f.lower().endswith(supported_types)]), config["num_repeats"])}

  for folder in folders:
    if not os.path.exists(folder):
      print(f"💥 Error: The folder {folder} doesn't exist.")
      return
  for folder, (img, rep) in images_repeats.items():
    if not img:
      print(f"💥 Error: Your {folder} folder is empty.")
      return
  for f in files:
    if not f.lower().endswith(".txt") and not f.lower().endswith(supported_types):
      print(f"💥 Error: Invalid file in dataset: \"{f}\". Aborting.")
      return

  pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
  steps_per_epoch = pre_steps_per_epoch/config["train_batch_size"]
  total_steps = int(config["epochs"]*steps_per_epoch)
  estimated_epochs = int(total_steps/steps_per_epoch)

  for folder, (img, rep) in images_repeats.items():
    print("📁"+folder + (" (Regularization)" if folder in reg else ""))
    print(f"📈 Found {img} images with {rep} repeats, equaling {img*rep} steps.")
  print(f"📉 Divide {pre_steps_per_epoch} steps by {config['train_batch_size']} batch size to get {steps_per_epoch} steps per epoch.")
  print(f"🔮 There will be {total_steps} steps, divided into {estimated_epochs} epochs and then some.")

  if total_steps > 10000:
    print("💥 Error: Your total steps are too high. You probably made a mistake. Aborting...")
    return

  return True

def create_train_config(config):
  config_dict = {
    "additional_network_arguments": {
      "unet_lr": config["unet_learning_rate"],
      "text_encoder_lr": config["te_learning_rate"],
      "network_dim": 32, # Important
      "network_alpha": 8, # Important
      "network_module": "networks.lora",
      "network_args": None,
      "network_train_unet_only": False,
      "network_weights": None
    },
    "optimizer_arguments": {
      "learning_rate": 5e-4, # Important
      "lr_scheduler": config["scheduler"], # Important
      "lr_scheduler_num_cycles": 0 if config["scheduler"] == "prodigy" else 3,
      "lr_scheduler_power": 0,
      "lr_warmup_steps": 0,
      "optimizer_type": config["optimizer"] , # Important
      "optimizer_args": config["optimizer_args"]  
    },
    "training_arguments": {
      "max_train_steps": None,
      "max_train_epochs": config["epochs"], # Important
      "save_every_n_epochs": config["epochs"] // 2,
      "save_last_n_epochs": config["epochs"],
      "train_batch_size": 4,
      "noise_offset": None,
      "clip_skip": 2,
      "min_snr_gamma": 5.0,
      "weighted_captions": False,
      "seed": 42,
      "max_token_length": 225,
      "xformers": True,
      "lowram": False,
      "max_data_loader_n_workers": 8,
      "persistent_data_loader_workers": True,
      "save_precision": "fp16",
      "mixed_precision": "fp16",
      "output_dir": config["output_folder"],
      "logging_dir": config["log_folder"],
      "output_name": config['project_name'],
      "log_prefix": config['project_name'],
    },
    "model_arguments": {
      "pretrained_model_name_or_path": config['model_file'],
      "v2": config['custom_model_is_based_on_sd2'],
      "v_parameterization": False
    },
    "saving_arguments": {
      "save_model_as": "safetensors",
    },
    "dreambooth_arguments": {
      "prior_loss_weight": 1.0,
    },
    "dataset_arguments": {
      "cache_latents": True,
    },
  }

  for key in config_dict:
    if isinstance(config_dict[key], dict):
      config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

  with open(config["config_file"], "w") as f:
    f.write(toml.dumps(config_dict))
  print(f"\n📄 Config saved to {config['config_file']}")

  dataset_config_dict = {
    "general": {
      "resolution": 512,
      "shuffle_caption": True,
      "keep_tokens": config["keep_tokens"],
      "flip_aug": True,
      "caption_extension": "",
      "enable_bucket": True,
      "bucket_reso_steps": 64,
      "bucket_no_upscale": False,
      "min_bucket_reso": 256,
      "max_bucket_reso": 1024,
    },
      "datasets": [
        {
          "subsets": [
            {
              "num_repeats": config["num_repeats"],
              "image_dir": config["images_folder"],
              "class_tokens": None
            }
          ]
        }
      ]
    }

  for key in dataset_config_dict:
    if isinstance(dataset_config_dict[key], dict):
      dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

  with open(config["dataset_config_file"], "w") as f:
    f.write(toml.dumps(dataset_config_dict))
  print(f"📄 Dataset config saved to {config['dataset_config_file']}")

def download_model(config):
  real_model_url = config["model_url"].strip()

  if real_model_url.lower().endswith((".ckpt", ".safetensors")):
    model_file = os.path.join(config["models_folder"], real_model_url[real_model_url.rfind('/'):])
  else:
    model_file = os.path.join(config["models_folder"], "downloaded_model.safetensors")
    if os.path.exists(model_file):
      os.system(f'rm "{model_file}"')
  if m := re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", config["model_url"]):
    real_model_url = real_model_url.replace("blob", "resolve")
  elif m := re.search(r"(?:https?://)?(?:www\\.)?civitai\.com/models/([0-9]+)(/[A-Za-z0-9-_]+)?", config["model_url"]):
    if m.group(2):
      model_file = os.path.join(config["models_folder"], f"{m.group(2)}.safetensors")
    if m := re.search(r"modelVersionId=([0-9]+)", config["model_url"]):
      real_model_url = f"https://civitai.com/api/download/models/{m.group(1)}"
    else:
      raise ValueError("optional_custom_training_model_url contains a civitai link, but the link doesn't include a modelVersionId. You can also right click the download button to copy the direct download link.")
  
  os.system(f'aria2c "{real_model_url}" --console-log-level=warn -c -s 16 -x 16 -k 10M -d / -o "{model_file}"')

  if model_file.lower().endswith(".safetensors"):
    from safetensors.torch import load_file as load_safetensors
    try:
      test = load_safetensors(model_file)
      del test
    except Exception as e:
      #if "HeaderTooLarge" in str(e):
      print(e)
      new_model_file = os.path.splitext(model_file)[0]+".ckpt"
      os.system(f'mv "{model_file}" "{new_model_file}"')
      model_file = new_model_file
      print(f"Renamed model to {os.path.splitext(model_file)[0]}.ckpt")

  if model_file.lower().endswith(".ckpt"):
    from torch import load as load_ckpt
    try:
      test = load_ckpt(model_file)
      del test
    except Exception as e:
      return False
  
  config["model_file"] = model_file
  return True

def main(is_install_dependencies, lora_name, output):
  config = get_config(lora_name, is_install_dependencies)
  if output:
    config["output_folder"] = output

  for dir in (config["models_folder"], config["deps_dir"], config["kohya_trainer_repo_dir"], config["log_folder"], config["images_folder"], config["output_folder"], config["config_folder"]):
    os.makedirs(dir, exist_ok=True)

  if not download_dataset(config["GCS_BUCKET_NAME"], config["project_name"], config["images_folder"]):
    print("\n💥 Error: Dataset does not exist in Google Cloud Storage bucket.")
    return
  if not validate_dataset(config):
    return

  if is_install_dependencies:
    print("\n🏭 Installing dependencies...\n")
    t0 = time()
    install_dependencies(config)
    t1 = time()
    print(f"\n✅ Installation finished in {int(t1-t0)} seconds.")
  else:
    print("\n✅ Skipping dependency installation.")
  
  if not (os.path.exists(config["models_folder"]) and os.listdir(config["models_folder"])):
    print("\n🔄 Downloading model...")
    if not download_model(config):
      print("\n💥 Error: The model you selected is invalid or corrupted, or couldn't be downloaded. You can use a civitai or huggingface link, or any direct download link.")
      return
  else:
    model_file = os.path.join(config["models_folder"], os.listdir(config["models_folder"])[0])
    config["model_file"] = model_file
    print("\n🔄 Model already downloaded.\n")
  
  create_train_config(config)
  
  print("\n⭐ Starting trainer...\n")
  os.chdir(config["kohya_trainer_repo_dir"])
  os.system(f'/bin/bash -c "source ../../.venv/bin/activate && accelerate launch --config_file={config["accelerate_config_file"]} --num_cpu_threads_per_process=2 train_network.py --dataset_config={config["dataset_config_file"]} --config_file={config["config_file"]}"')
  print("Done!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--install_dependencies", type=bool, default=False, help="Install dependencies before training")
  parser.add_argument("--lora_name", type=str, default="lora", help="Name of the LORA to be trained")
  parser.add_argument("--output", type=str, default="", help="Output directory for the trained checkpoint")
  args = parser.parse_args()
  
  main(is_install_dependencies=args.install_dependencies, lora_name=args.lora_name, output=args.output)
