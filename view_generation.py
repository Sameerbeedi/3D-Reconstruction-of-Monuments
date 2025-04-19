import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import HEDdetector, MLSDdetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MiDaSDetector, OpenposeDetector
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download # <-- Changed from cached_download

# --- Configuration ---
# Choose the appropriate ControlNet model based on desired view generation method
# Depth seems suitable for generating different views from a single image
CONTROLNET_MODEL_ID = "lllyasviel/control_v11f1p_sd15_depth"
STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5" # Base model for ControlNet

# --- Helper Functions ---

def load_controlnet_pipeline(device='cuda'):
    """Loads the ControlNet pipeline for depth-controlled image generation."""
    print("Loading ControlNet pipeline...")
    try:
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            STABLE_DIFFUSION_MODEL_ID, controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # Remove following line if using CPU or have enough VRAM
        # pipe.enable_model_cpu_offload() # Offload parts to CPU if VRAM is limited
        pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention() # Use if xformers is installed
        print("ControlNet pipeline loaded.")
        return pipe
    except Exception as e:
        print(f"Error loading ControlNet pipeline: {e}")
        print("Ensure you have the necessary libraries installed and model IDs are correct.")
        return None

def get_depth_map(image, device='cuda'):
    """Generates a depth map for the input image."""
    print("Generating depth map...")
    try:
        # Initialize the MiDaS depth estimator
        depth_estimator = MiDaSDetector.from_pretrained("Intel/dpt-hybrid-midas")
        depth_map_image = depth_estimator(image, detect_resolution=384, image_resolution=512) # Adjust resolutions as needed
        print("Depth map generated.")
        return depth_map_image
    except Exception as e:
        print(f"Error generating depth map: {e}")
        return None

# --- Main Function for incorporated.py ---

def generate_multiple_views(input_image_path, num_views=4, output_dir="hampi_output/views", device='cuda', prompt_prefix="Another view of the Hampi monument"):
    """Generates multiple views of an object from a single input image using ControlNet."""
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}")
        return []

    # Load models
    controlnet_pipe = load_controlnet_pipeline(device=device)
    if not controlnet_pipe:
        return []

    # Prepare output directory
    view_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(input_image_path))[0] + "_views")
    os.makedirs(view_output_dir, exist_ok=True)
    print(f"Saving generated views to: {view_output_dir}")

    # Load input image
    try:
        input_image = Image.open(input_image_path).convert("RGB")
        # Optional: Resize image if needed for consistency or performance
        # input_image = input_image.resize((512, 512))
    except Exception as e:
        print(f"Error loading input image {input_image_path}: {e}")
        return []

    # Get depth map (Control Image)
    control_image = get_depth_map(input_image, device=device)
    if not control_image:
        return []

    # Define base prompt and negative prompt
    # You might want to extract details from the filename or use a fixed prompt
    base_prompt = f"{prompt_prefix}, realistic photo, ancient stone architecture, Hampi, India, detailed stonework, clear daylight, high resolution"
    negative_prompt = "cartoon, illustration, anime, 3d render, painting, sketch, drawing, blur, distortion, low quality, poor lighting, oversaturated, fantasy elements, text, words, signature, watermark"

    generated_image_paths = []

    print(f"Generating {num_views} different views...")
    for i in tqdm(range(num_views)):
        # Generate image with ControlNet
        # You can slightly vary the prompt or seed for different views
        # For more distinct views, you might need more sophisticated techniques
        # like manipulating the depth map or using different ControlNets (e.g., Normal maps, Canny edges)
        # or dedicated multi-view models if available.
        # Here, we rely on the inherent randomness of the diffusion process with a fixed control.
        try:
            generator = torch.Generator(device=device).manual_seed(i * 1234 + 5678) # Vary seed for variation
            output_image = controlnet_pipe(
                prompt=base_prompt,
                negative_prompt=negative_prompt,
                image=control_image, # Provide the depth map as the control
                num_inference_steps=30, # Adjust steps (20-50 typical)
                guidance_scale=7.5,     # Adjust guidance
                generator=generator
            ).images[0]

            # Save the generated image
            filename = f"view_{i+1}.png"
            save_path = os.path.join(view_output_dir, filename)
            output_image.save(save_path)
            generated_image_paths.append(save_path)
            print(f"Saved view {i+1} to {save_path}")

        except Exception as e:
            print(f"Error generating view {i+1}: {e}")

    print(f"Generated {len(generated_image_paths)} views.")
    return generated_image_paths


# Example usage (optional)
if __name__ == "__main__":
    print("Running view_generation.py directly for testing...")
    # Create a dummy input image file for testing if needed
    # dummy_image_path = "dummy_input.png"
    # if not os.path.exists(dummy_image_path):
    #     Image.new('RGB', (512, 512), color = 'red').save(dummy_image_path)

    # test_input_image = dummy_image_path # Replace with a real image path
    # if os.path.exists(test_input_image):
    #     generate_multiple_views(
    #         input_image_path=test_input_image,
    #         num_views=2,
    #         output_dir="test_hampi_output/views",
    #         device='cuda' if torch.cuda.is_available() else 'cpu'
    #     )
    # else:
    #     print(f"Test input image '{test_input_image}' not found.")
    pass