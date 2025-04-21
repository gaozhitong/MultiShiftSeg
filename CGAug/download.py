
import gdown
import os
from huggingface_hub import hf_hub_download

# Download SAM
print("Downloading SAM...")
sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
download_dir = "./pretrained_model"

if not os.path.exists(download_dir):
    os.makedirs(download_dir)
if not os.path.isfile(os.path.join(download_dir, "sam_vit_h_4b8939.pth")):
    sam_path = os.path.join(download_dir, "sam_vit_h_4b8939.pth")
    os.system(f"wget {sam_url} -O {sam_path}")

# Download anomaly detector
print("Downloading anomaly detector...")

anomaly_url = "https://drive.google.com/uc?id=1UVms08chnBkZta_cNumjiei6GByyM9VN"
anomaly_path = os.path.join(download_dir, "bt-f-xl.pth")
if not os.path.isfile(anomaly_path):
    gdown.download(anomaly_url, anomaly_path, quiet=False)

# Download controlnet
print("Downloading controlnet...")
print("Note that there is no verbose output for this download. Please be patient :)")
controlnet_name = "/models/control_sd15_seg.pth"
controlnet_path = os.path.join(download_dir, "control_sd15_seg.pth")
if not os.path.isfile(controlnet_path):
    hf_hub_download("lllyasviel/ControlNet",
                    filename="models/control_sd15_seg.pth",
                    local_dir=download_dir,)
    os.system(f"mv {download_dir}/models/control_sd15_seg.pth {controlnet_path}")

