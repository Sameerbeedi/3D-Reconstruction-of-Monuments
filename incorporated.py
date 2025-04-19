import argparse
import os
import sys
import random
from datetime import datetime
import torch
import re # <-- Add this import

# --- Import Pipeline Components ---
print("Importing pipeline components...")
try:
    from qa_to_image_pipeline import qa_to_images_pipeline
    print("- Successfully imported: Image Generation Pipeline")
except ImportError as e:
    print(f"Warning: Could not import from qa_to_image_pipeline.py: {e}")
    qa_to_images_pipeline = None

try:
    from store_utils import process_architectural_qa_batch_and_save, search_architecture_qa_database, DEFAULT_QA_TXT_FILE, DEFAULT_ARCH_QA_DB_PATH
    print("- Successfully imported: Q&A Generation & Search Utilities")
except ImportError as e:
    print(f"Warning: Could not import from store_utils.py: {e}")
    process_architectural_qa_batch_and_save = None
    search_architecture_qa_database = None
    # Define defaults here if import fails, to avoid NameError later
    DEFAULT_QA_TXT_FILE = "Hampi_Architecture_QA.txt"
    DEFAULT_ARCH_QA_DB_PATH = "./chroma_architecture_qa_db"


try:
    from view_generation import generate_multiple_views
    print("- Successfully imported: 3D View Generation Utilities")
except ImportError as e:
    print(f"Warning: Could not import from view_generation.py: {e}")
    generate_multiple_views = None

# Placeholder for rag_utils if needed for other modes in the future
# try:
#     from rag_utils import search_general_text_database
#     print("- Successfully imported: General RAG Utilities")
# except ImportError as e:
#     print(f"Warning: Could not import from rag_utils.py: {e}")
#     search_general_text_database = None
print("Imports complete.")
# --- End Imports ---


# Set up command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Hampi Monument Research Pipeline")

    # Main operation mode
    parser.add_argument("--mode", type=str, required=True, choices=[
        "generate_qa", "search_qa", "generate_images", "generate_views", "full_pipeline"
    ], help="Operation mode")

    # store_utils options
    parser.add_argument("--num_questions", type=int, default=10,
                      help="Number of Q&A pairs to generate (for mode=generate_qa or full_pipeline)")
    parser.add_argument("--qa_topic", type=str, default=None, # Added back for targeted generation
                      help="Specific architectural topic to focus on (for mode=generate_qa)")
    parser.add_argument("--qa_output_file", type=str, default=DEFAULT_QA_TXT_FILE,
                      help="Output file path for generated Q&A text")
    parser.add_argument("--arch_qa_db", type=str, default=DEFAULT_ARCH_QA_DB_PATH,
                      help="Path to the architectural Q&A vector database")

    # Search / RAG options
    parser.add_argument("--query", type=str, default=None,
                      help="Query for searching the Q&A database (mode=search_qa) or for vector store image generation")

    # Image generation options (qa_to_image_pipeline)
    parser.add_argument("--qa_source", type=str, choices=["file", "vector_store"], default="file",
                      help="Source of Q&A data for image generation: text file or vector database")
    # Note: --qa_file argument renamed to --qa_input_file to avoid conflict with --qa_output_file
    parser.add_argument("--qa_input_file", type=str, default=DEFAULT_QA_TXT_FILE,
                      help="Path to Q&A text file (if qa_source=file)")
    parser.add_argument("--num_images", type=int, default=1,
                      help="Number of images to generate per Q&A pair")
    parser.add_argument("--output_dir", type=str, default="hampi_output",
                      help="Main directory to save generated outputs (images, views, Q&A txt)")
    parser.add_argument("--monuments", type=str, nargs='+', default=None,
                      help="Filter image generation for specific monuments mentioned in Q&A")

    # 3D view generation options (view_generation)
    parser.add_argument("--input_image", type=str, default=None,
                      help="Input image for 3D view generation (for mode=generate_views)")
    parser.add_argument("--num_views", type=int, default=4,
                      help="Number of different views to generate (for mode=generate_views or full_pipeline)")

    # General options
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device for computation: 'cuda' or 'cpu'")

    return parser.parse_args()


# Main execution logic
def main():
    args = parse_args()
    # Ensure the main output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'='*15} Starting Hampi Pipeline {'='*15}")
    print(f"Operation Mode: {args.mode}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Using Device: {args.device}")
    print(f"{'='*50}\n")


    start_time = datetime.now()
    generated_image_paths = [] # To store paths for potential use in full_pipeline

    # --- Mode Execution ---
    if args.mode == "generate_qa":
        if process_architectural_qa_batch_and_save:
            print(f"Generating {args.num_questions} architectural Q&A pairs...")
            # Construct full path for output file within the main output directory
            qa_output_filepath = os.path.join(args.output_dir, os.path.basename(args.qa_output_file))
            process_architectural_qa_batch_and_save(
                num_questions=args.num_questions,
                arch_qa_db_path=args.arch_qa_db,
                qa_txt_file=qa_output_filepath,
                topic=args.qa_topic # Pass the topic if specified
            )
        else:
            print("Error: Q&A generation function (process_architectural_qa_batch_and_save) not available. Please check imports and store_utils.py.")

    elif args.mode == "search_qa":
        if search_architecture_qa_database and args.query:
            print(f"Searching architectural Q&A database for: '{args.query}'")
            results = search_architecture_qa_database(
                query=args.query,
                arch_qa_db_path=args.arch_qa_db
            )
            print("\nSearch Results:")
            if results:
                for i, doc in enumerate(results):
                    print(f"\n--- Result {i+1} ---")
                    # Extract Q&A from the stored text
                    content = doc.page_content
                    q_match = re.search(r'Question: (.*?)(?=\nAnswer:|$)', content)
                    a_match = re.search(r'Answer: (.*?)(?=$)', content, re.DOTALL)
                    if q_match: print(f"Q: {q_match.group(1).strip()}")
                    if a_match: print(f"A: {a_match.group(1).strip()}")
                    print(f"(Similarity Score: {doc.metadata.get('_distance', 'N/A')})") # If distance is available
                    print("----------")
            else:
                print("No relevant results found in the architectural Q&A database.")
        elif not args.query:
             print("Error: Please provide a query using --query for search_qa mode.")
        else:
            print("Error: Q&A search function (search_architecture_qa_database) not available. Please check imports and store_utils.py.")

    elif args.mode == "generate_images":
        if qa_to_images_pipeline:
            print("Starting image generation pipeline...")
            # Construct full path for input file if using file source
            qa_input_filepath = os.path.join(args.output_dir, os.path.basename(args.qa_input_file)) if args.qa_source == 'file' else args.qa_input_file

            generated_image_paths = qa_to_images_pipeline(
                source_type=args.qa_source,
                file_path=qa_input_filepath, # Use constructed path
                vector_store_path=args.arch_qa_db, # Use the arch QA db path
                query=args.query, # Pass query if source is vector_store
                num_images=args.num_images,
                output_dir=args.output_dir, # Save images in the main output dir
                monument_filters=args.monuments
            )
        else:
            print("Error: Image generation function (qa_to_images_pipeline) not available. Please check imports and qa_to_image_pipeline.py.")

    elif args.mode == "generate_views":
        if generate_multiple_views and args.input_image:
            print(f"Generating {args.num_views} views for image: {args.input_image}")
            if not os.path.exists(args.input_image):
                 print(f"Error: Input image not found at {args.input_image}")
            else:
                # Views will be saved in a subdirectory within output_dir by the function
                generate_multiple_views(
                    input_image_path=args.input_image,
                    num_views=args.num_views,
                    output_dir=args.output_dir, # Pass the main output dir
                    device=args.device
                )
        elif not args.input_image:
             print("Error: Please provide an input image using --input_image for generate_views mode.")
        else:
            print("Error: 3D view generation function (generate_multiple_views) not available. Please check imports and view_generation.py.")

    elif args.mode == "full_pipeline":
        print("Running full pipeline...")
        qa_file_for_images = os.path.join(args.output_dir, os.path.basename(args.qa_output_file)) # Default path

        # Step 1: Generate Q&A
        if process_architectural_qa_batch_and_save:
            print(f"\n--- Step 1: Generating {args.num_questions} Q&A pairs ---")
            process_architectural_qa_batch_and_save(
                num_questions=args.num_questions,
                arch_qa_db_path=args.arch_qa_db,
                qa_txt_file=qa_file_for_images, # Save to output dir
                topic=args.qa_topic
            )
            print("--- Q&A Generation Complete ---")
        else:
            print("\nWarning: Q&A generation function not available. Skipping Step 1.")
            # Check if the specified input file exists if we skip generation
            if not os.path.exists(qa_file_for_images):
                 print(f"Error: Q&A input file '{qa_file_for_images}' not found and generation was skipped. Cannot proceed.")
                 sys.exit(1) # Exit if we can't generate or find the input for the next step


        # Step 2: Generate Images from Q&A
        if qa_to_images_pipeline:
            print(f"\n--- Step 2: Generating {args.num_images} images per Q&A pair ---")
            generated_image_paths = qa_to_images_pipeline(
                source_type="file", # Use the file generated/specified
                file_path=qa_file_for_images,
                num_images=args.num_images,
                output_dir=args.output_dir,
                monument_filters=args.monuments
            )
            print("--- Image Generation Complete ---")
        else:
            print("\nWarning: Image generation function not available. Skipping Step 2.")


        # Step 3: Generate Views for a random generated image (example)
        if generate_multiple_views and generated_image_paths:
            # Select one of the generated images randomly
            image_to_process = random.choice(generated_image_paths)
            print(f"\n--- Step 3: Generating {args.num_views} views for a generated image: {os.path.basename(image_to_process)} ---")
            generate_multiple_views(
                input_image_path=image_to_process,
                num_views=args.num_views,
                output_dir=args.output_dir, # Views saved in subdirectory here
                device=args.device
            )
            print("--- View Generation Complete ---")
        elif not generated_image_paths:
             print("\nSkipping view generation (Step 3) as no images were generated in Step 2.")
        else:
            print("\nWarning: 3D view generation function not available. Skipping Step 3.")

    else:
        # This case should not be reachable due to argparse choices
        print(f"Error: Unknown mode '{args.mode}'.")


    end_time = datetime.now()
    print(f"\n{'='*15} Pipeline Finished {'='*15}")
    print(f"Total execution time: {end_time - start_time}")
    print(f"{'='*50}\n")

# Entry point
if __name__ == "__main__":
    main()