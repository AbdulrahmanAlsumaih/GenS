import os
import torch
import numpy as np
from PIL import Image
import sys
import json

# Add the project directory to Python path
# Adjust this path to the root of your project directory
sys.path.append('/path/to/your/project')

# Import your model
from models.gens import GenS

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_str = f.read()
    # Convert HOCON-like format to JSON
    config_json = '{' + config_str.replace('=', ':').replace('{', '{').replace('}', '}') + '}'
    return json.loads(config_json)

def load_model(config, checkpoint_path, device):
    model = GenS(config["model"]).to(device)
    model.load_params_vol(checkpoint_path, device)
    model.eval()
    return model, config

def prepare_input(image_path, intrinsics, c2w, img_hw):
    # Load and preprocess image
    image = Image.open(image_path).resize(img_hw[::-1])
    image = np.array(image) / 255.0
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    
    # Prepare other inputs
    intrinsics = torch.from_numpy(intrinsics).float().unsqueeze(0)
    c2w = torch.from_numpy(c2w).float().unsqueeze(0)
    
    return {
        "imgs": image,
        "intrs": intrinsics,
        "c2ws": c2w,
    }

def run_inference(model, inputs):
    with torch.no_grad():
        outputs = model("val", inputs, cos_anneal_ratio=1.0)
    return outputs

def save_outputs(outputs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save rendered image
    img_fine = outputs["img_fine"].cpu().numpy()[0]
    Image.fromarray((img_fine * 255).astype(np.uint8)).save(os.path.join(output_dir, 'rendered_image.png'))
    
    # Save depth map
    depth = outputs["render_depth"].cpu().numpy()[0]
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    Image.fromarray((depth_normalized * 255).astype(np.uint8)).save(os.path.join(output_dir, 'depth_map.png'))
    
    # Save normal map
    normal_img = outputs["normal_img"].cpu().numpy()[0]
    Image.fromarray((normal_img * 255).astype(np.uint8)).save(os.path.join(output_dir, 'normal_map.png'))

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Set paths (adjust these to your actual file locations)
    config_path = "/content/GenS/confs/gens_bmvs.conf"
    checkpoint_path = "/content/model_049.ckpt"
    image_path = "/content/input_image.png"
    output_dir = "/content/GenS/output"

    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)

    # Load model and config
    print("Loading model and config...")
    model, conf = load_model(config, checkpoint_path, device)

    # Get image size from config
    img_hw = conf["val_dataset"]["img_hw"]

    # Prepare input
    print("Preparing input...")
    intrinsics = np.eye(4)  # Replace with actual intrinsics
    c2w = np.eye(4)  # Replace with actual camera-to-world matrix
    inputs = prepare_input(image_path, intrinsics, c2w, img_hw)

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    print("Running inference...")
    outputs = run_inference(model, inputs)

    # Save outputs
    print("Saving outputs...")
    save_outputs(outputs, output_dir)

    print(f"Inference complete. Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()