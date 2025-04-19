import os
import re
from PIL import Image
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler # <-- Import added here
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download # <-- Changed from cached_download if it was here
import time

# --- Configuration ---
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Load required models
def load_models(device="cpu"):
    """Loads the text processing and image generation models."""
    print("Loading models...")
    # Load text processing model for enhancing architectural descriptions
    text_processor = pipeline("text-generation", model="gpt2-large")
    
    # Load image generation model (example using Stable Diffusion)
    image_generator = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16
    )
    try:
        # Load Stable Diffusion Pipeline
        model_id = "stabilityai/stable-diffusion-2-1-base" # Or another SD model
        image_generator = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32) # Use float32 for CPU
    
        # Set a better scheduler (optional but often recommended)
        image_generator.scheduler = EulerAncestralDiscreteScheduler.from_config(
            image_generator.scheduler.config
        )
        image_generator = image_generator.to(device)
        print(f"- Image generation model ({model_id}) loaded to {device}.")
    except Exception as e:
        print(f"Error loading image generation model: {e}")
        image_generator = None

    return text_processor, image_generator

# Extract architectural details from Q&A pairs
def extract_architectural_details(qa_text):
    """
    Parse Q&A text to extract key architectural details
    that can be used in image generation prompts
    """
    # Extract answer portion from Q&A text
    answer_match = re.search(r'Answer: (.*?)(?=$|\n\n)', qa_text, re.DOTALL)
    if not answer_match:
        return None
        
    answer_text = answer_match.group(1).strip()
    
    # Extract key visual elements using simple heuristics
    # Look for sentences with visual descriptors
    visual_sentences = []
    
    # Keywords that suggest visual elements
    visual_keywords = [
        'feature', 'design', 'decorated', 'carved', 'ornate', 'structure', 
        'shape', 'pattern', 'motif', 'pillar', 'column', 'arch', 'dome',
        'appearance', 'visible', 'style', 'height', 'proportion', 'material'
    ]
    
    # Extract sentences that contain visual elements
    sentences = re.split(r'(?<=[.!?])\s+', answer_text)
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in visual_keywords):
            visual_sentences.append(sentence)
    
    # Join the visual elements together
    if visual_sentences:
        return " ".join(visual_sentences)
    else:
        # If no specific visual elements found, use the first 3 sentences
        return " ".join(sentences[:3])

# Format architectural details into optimized image generation prompts
def format_image_prompt(architectural_details, monument_name=None):
    """
    Create a well-structured prompt for image generation
    that emphasizes realism and architectural accuracy
    """
    # Base prompt components for realistic Hampi architecture
    base_prompt = [
        "realistic photo of historical architecture",
        "ancient stone carving",
        "archaeological site",
        "16th century Vijayanagara Empire architecture",
        "UNESCO world heritage site",
        "detailed stonework",
        "documentary photography style",
        "photorealistic rendering",
        "clear daylight",
        "high resolution"
    ]
    
    # Add monument name if provided
    if monument_name:
        specific_prompt = f"The {monument_name} at Hampi, India. {architectural_details}"
    else:
        specific_prompt = f"A monument at Hampi, India. {architectural_details}"
    
    # Combine components into final prompt
    final_prompt = specific_prompt + " " + ", ".join(base_prompt)
    
    # Add negative prompts for image generation parameters
    negative_prompt = "cartoon, illustration, anime, 3d render, painting, sketch, drawing, blur, distortion, low quality, poor lighting, oversaturated, fantasy elements"
    
    return final_prompt, negative_prompt

# Generate images based on architectural descriptions
def generate_images(prompt, negative_prompt, image_generator, num_images=1, output_dir="hampi_images"):
    """
    Generate realistic images of Hampi architecture based on prompts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images
    images = image_generator(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=50,  # Higher for better quality
        guidance_scale=7.5       # Control adherence to prompt
    ).images
    
    # Save images
    image_paths = []
    for i, image in enumerate(images):
        # Create filename based on shortened prompt
        short_desc = prompt.split('.')[0][:30].replace(" ", "_")
        filename = f"{short_desc}_{i}.png"
        save_path = os.path.join(output_dir, filename)
        
        # Save the image
        image.save(save_path)
        image_paths.append(save_path)
        print(f"Saved image to {save_path}")
    
    return image_paths

# Extract Q&A pairs from vector store or text file
def get_qa_pairs(source_type="file", file_path="Hampi_Architecture_QA.txt", vector_store_path=None, query=None):
    """
    Extract Q&A pairs from either:
    1. Text file generated by store.ipynb
    2. Vector store created by store.ipynb
    """
    qa_pairs = []
    
    if source_type == "file":
        # Parse the text file format
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract Q&A blocks
            blocks = re.split(r'-{40,}', content)
            for block in blocks:
                if not block.strip():
                    continue
                
                # Extract question and answer
                q_match = re.search(r'Q\d+: (.*?)(?=\n|$)', block)
                a_match = re.search(r'A\d+: (.*?)(?=\n|$)', block, re.DOTALL)
                
                if q_match and a_match:
                    qa_pairs.append({
                        "question": q_match.group(1).strip(),
                        "answer": a_match.group(1).strip(),
                    })
        else:
            print(f"File not found: {file_path}")
    
    elif source_type == "vector_store" and vector_store_path and query:
        # Load the vector store
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embedding_model)
        
        # Search for relevant Q&A pairs
        results = vector_store.similarity_search(query, k=5)
        
        for doc in results:
            # Extract question and answer from the document content
            content = doc.page_content
            q_match = re.search(r'Question: (.*?)(?=\nAnswer:|$)', content)
            a_match = re.search(r'Answer: (.*?)(?=$)', content, re.DOTALL)
            
            if q_match and a_match:
                qa_pairs.append({
                    "question": q_match.group(1).strip(),
                    "answer": a_match.group(1).strip(),
                })
    
    return qa_pairs

# Main pipeline function
def qa_to_images_pipeline(source_type="file", file_path="Hampi_Architecture_QA.txt", 
                         vector_store_path="./chroma_architecture_qa_db", query=None,
                         num_images=1, output_dir="hampi_images", monument_filters=None):
    """
    Complete pipeline to convert architectural Q&A data to realistic images
    
    Args:
        source_type: 'file' or 'vector_store'
        file_path: Path to the QA text file (if source_type is 'file')
        vector_store_path: Path to the vector store (if source_type is 'vector_store')
        query: Search query for the vector store (if source_type is 'vector_store')
        num_images: Number of images to generate per QA pair
        output_dir: Directory to save the generated images
        monument_filters: List of specific monuments to include (e.g., ['Vitthala Temple'])
    """
    print("Loading models...")
    text_processor, image_generator = load_models()
    
    print("Retrieving Q&A pairs...")
    qa_pairs = get_qa_pairs(source_type, file_path, vector_store_path, query)
    
    if not qa_pairs:
        print("No Q&A pairs found!")
        return []
    
    print(f"Found {len(qa_pairs)} Q&A pairs.")
    
    # Filter by monument if specified
    if monument_filters:
        filtered_pairs = []
        for pair in qa_pairs:
            if any(monument.lower() in pair["question"].lower() for monument in monument_filters):
                filtered_pairs.append(pair)
        qa_pairs = filtered_pairs
        print(f"Filtered to {len(qa_pairs)} Q&A pairs related to specified monuments.")
    
    generated_image_paths = []
    
    for i, pair in enumerate(qa_pairs):
        print(f"\nProcessing Q&A pair {i+1}/{len(qa_pairs)}")
        print(f"Question: {pair['question']}")
        
        # Extract monument name from question if possible
        monument_match = re.search(r'(Vitthala|Virupaksha|Krishna|Hazara Rama|Lotus Mahal|Elephant Stables)', pair['question'])
        monument_name = monument_match.group(1) if monument_match else None
        
        # Extract architectural details
        full_qa_text = f"Question: {pair['question']}\nAnswer: {pair['answer']}"
        architectural_details = extract_architectural_details(full_qa_text)
        
        if not architectural_details:
            print("Could not extract architectural details, skipping...")
            continue
        
        print(f"Extracted details: {architectural_details[:100]}...")
        
        # Format image generation prompt
        prompt, negative_prompt = format_image_prompt(architectural_details, monument_name)
        print(f"Generated prompt: {prompt[:100]}...")
        
        # Generate images
        image_paths = generate_images(prompt, negative_prompt, image_generator, num_images, output_dir)
        generated_image_paths.extend(image_paths)
    
    print(f"\nGeneration complete. Created {len(generated_image_paths)} images in {output_dir}.")
    return generated_image_paths

# Example usage
if __name__ == "__main__":
    # Example 1: Generate images from text file
    # qa_to_images_pipeline(
    #     source_type="file",
    #     file_path="Hampi_Architecture_QA.txt",
    #     num_images=2,
    #     monument_filters=["Vitthala Temple", "pillars"]
    # )
    
    # Example 2: Generate images from vector store with query
    # qa_to_images_pipeline(
    #     source_type="vector_store",
    #     vector_store_path="./chroma_architecture_qa_db",
    #     query="What are the distinctive features of pillars in Hampi temples?",
    #     num_images=2
    # )
    pass


# how to use:

# Generate Q&A Data: First run the system in generate_qa mode to create architectural knowledge
# python pipeline_integration_script.py --mode generate_qa --num_questions 20

# Generate Images: Use the Q&A data to create realistic images
# python pipeline_integration_script.py --mode generate_images --qa_source file --qa_file Hampi_Architecture_QA.txt --num_images 2 --monuments "Vitthala Temple" "Krishna Temple"

# Generate 3D Views: Create multiple perspectives of the same monument
# python pipeline_integration_script.py --mode generate_views --input_image hampi_output/vitthala_temple_0.png --num_views 4

# Run the Full Pipeline: Do everything in one command
# python pipeline_integration_script.py --mode full_pipeline --num_questions 10 --num_images 3 --num_views 2