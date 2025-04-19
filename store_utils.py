import os
import time
import uuid
import re
import random # Added import for random.choice
# Updated imports for LangChain components
from langchain_community.vectorstores import Chroma # Changed from langchain.vectorstores
from langchain_huggingface import HuggingFaceEmbeddings # Changed from langchain_community.embeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, SystemMessage
import traceback # Add traceback for detailed error printing

# --- Configuration ---
# Define default paths (can be overridden by arguments if needed)
DEFAULT_SOURCE_DB_PATH = "./chroma_db" # Assuming this is the source text DB
DEFAULT_ARCH_QA_DB_PATH = "./chroma_architecture_qa_db"
DEFAULT_QA_TXT_FILE = "Hampi_Architecture_QA.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192"
# Ensure you have your Groq API key set as an environment variable or replace the placeholder
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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") # Replace with your key if not using env var

# List of architectural topics (from store.ipynb)
hampi_architectural_topics = [
    "Vijayanagara architectural style", "Temple architecture", "Gopurams",
    "Mandapas (pillared halls)", "Pillars (musical pillars, ornate pillars)",
    "Islamic influences on Vijayanagara architecture", "Royal Enclosure structures",
    "Water structures (stepwells, tanks, aqueducts)", "Fortifications and gateways",
    "Bas-reliefs and carvings", "Materials used in construction (granite)",
    "Comparison with other South Indian styles (Dravidian, Chalukya)",
    "Specific monument features (e.g., Stone Chariot, Lotus Mahal design)"
]

# --- Helper Functions ---

def initialize_components(source_db_path=DEFAULT_SOURCE_DB_PATH, arch_qa_db_path=DEFAULT_ARCH_QA_DB_PATH):
    """Initializes embedding model, LLMs, and vector stores."""
    print("Initializing models and vector stores...")
    embedding_model, llm, question_generator_llm, source_vectorstore, architecture_qa_vectorstore, qa_chain = None, None, None, None, None, None # Initialize to None
    try:
        print("--> Initializing Embedding Model...")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("    Embedding Model Initialized.")

        # Source vector store (for context)
        print(f"--> Attempting to load source vector store from: {os.path.abspath(source_db_path)}")
        if not os.path.exists(source_db_path):
            print(f"    Error: Source DB directory not found: {os.path.abspath(source_db_path)}")
            raise FileNotFoundError(f"Directory not found: {source_db_path}")
        source_vectorstore = Chroma(persist_directory=source_db_path, embedding_function=embedding_model)
        print("    Source vector store loaded.")

        # Architecture Q&A vector store (for storing generated pairs)
        print(f"--> Attempting to load/create architecture Q&A vector store at: {os.path.abspath(arch_qa_db_path)}")
        architecture_qa_vectorstore = Chroma(persist_directory=arch_qa_db_path, embedding_function=embedding_model)
        print("    Architecture Q&A vector store loaded/initialized.")

        # LLMs
        print("--> Initializing LLMs...")
        # Define the placeholder key used as default
        placeholder_key = "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"
        # Modify the check: Error if key is missing OR if it's exactly the placeholder
        if not GROQ_API_KEY or GROQ_API_KEY == placeholder_key:
             print("    Error: GROQ_API_KEY is missing or is the default placeholder.")
             raise ValueError("Invalid or missing Groq API Key provided. Ensure it's set via environment variable.")
        llm = ChatGroq(model=LLM_MODEL, groq_api_key=GROQ_API_KEY)
        question_generator_llm = ChatGroq(model=LLM_MODEL, groq_api_key=GROQ_API_KEY) # Separate instance if needed
        print("    LLMs initialized.")

        # RAG chain for answering questions based on source documents
        print("--> Initializing RAG chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=source_vectorstore.as_retriever()
        )
        print("    RAG chain initialized.")

        print("--> Initialization complete (try block finished).")
        # Explicitly return the initialized components here
        return embedding_model, llm, question_generator_llm, source_vectorstore, architecture_qa_vectorstore, qa_chain

    except Exception as e:
        # Print the specific error that occurred during initialization
        print(f"\n!!! Error during initialization: {e} !!!\n")
        # Print the full traceback to see where the error originated
        traceback.print_exc()
        print("\n!!! Please ensure ChromaDB directories exist and are valid, required models are accessible, and API keys are valid. !!!\n")
        # Return None for all components to signal failure
        return None, None, None, None, None, None

def generate_architectural_question(question_generator_llm, topic=None):
    """Generates a specific architectural question about Hampi."""
    if not question_generator_llm:
        return "Error: Question generator LLM not initialized."

    # Use random.choice if topic is not provided
    selected_topic = topic if topic else random.choice(hampi_architectural_topics)

    prompt = f"""Generate a specific, detailed question about the architectural features of Hampi, focusing on the topic: '{selected_topic}'.
    Examples:
    - What are the typical dimensions and decorative motifs found on the pillars of the Vitthala Temple's main mandapa?
    - Describe the construction techniques used for the corbelled arches seen in the Lotus Mahal.
    - How does the design of the gopuram at the Virupaksha Temple incorporate elements from earlier Dravidian styles?

    Focus specifically on construction methods, design elements, structural innovations, or artistic aspects of the architecture.
    The question should concentrate on architectural style only, not history or cultural significance.
    Provide ONLY the question with no additional text."""

    messages = [
        SystemMessage(content="You are an architectural historian specializing in Hampi and Vijayanagara architecture."),
        HumanMessage(content=prompt)
    ]

    try:
        response = question_generator_llm.invoke(messages)
        # Basic cleaning: remove potential quotes or prefixes
        question = response.content.strip().strip('"').strip("'")
        if not question.endswith("?"):
             question += "?" # Ensure it's a question
        return question
    except Exception as e:
        print(f"Error generating question: {e}")
        return f"Could you detail the architectural features of {selected_topic} in Hampi?" # Fallback

def save_qa_to_txt(qa_pairs, filename=DEFAULT_QA_TXT_FILE):
    """Saves generated Q&A pairs to a text file."""
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write("Hampi Architectural Q&A\n")
            file.write("=" * 40 + "\n\n")

            for i, qa in enumerate(qa_pairs):
                file.write(f"Q{i+1}: {qa['question']}\n")
                file.write(f"A{i+1}: {qa['answer']}\n")
                file.write("-" * 40 + "\n\n")

        print(f"Q&A pairs saved to {filename}")
    except Exception as e:
        print(f"Error saving Q&A to file {filename}: {e}")


# --- Main Functions for incorporated.py ---

def process_architectural_qa_batch_and_save(num_questions=10, arch_qa_db_path=DEFAULT_ARCH_QA_DB_PATH, qa_txt_file=DEFAULT_QA_TXT_FILE, topic=None):
    """Generates and stores architecture-focused Q&A pairs."""
    embedding_model, llm, question_generator_llm, source_vectorstore, architecture_qa_vectorstore, qa_chain = initialize_components(arch_qa_db_path=arch_qa_db_path)

    # --- Existing Debugging ---
    print("\n--- Debugging component values after initialization ---")
    print(f"  embedding_model: {type(embedding_model)}, Is None: {embedding_model is None}")
    print(f"  llm: {type(llm)}, Is None: {llm is None}")
    print(f"  question_generator_llm: {type(question_generator_llm)}, Is None: {question_generator_llm is None}")
    print(f"  source_vectorstore: {type(source_vectorstore)}, Is None: {source_vectorstore is None}")
    print(f"  architecture_qa_vectorstore: {type(architecture_qa_vectorstore)}, Is None: {architecture_qa_vectorstore is None}")
    print(f"  qa_chain: {type(qa_chain)}, Is None: {qa_chain is None}")
    # Keep the all() print for comparison, but don't use it for the check
    print(f"  Result of all([...]): {all([embedding_model, llm, question_generator_llm, source_vectorstore, architecture_qa_vectorstore, qa_chain])}")
    print("--- End Debugging ---\n")
    # --- End Existing Debugging ---

    # --- Modified Check: Check each component individually ---
    initialization_failed = False
    if embedding_model is None:
        print("Initialization Check Failed: embedding_model is None")
        initialization_failed = True
    if llm is None:
        print("Initialization Check Failed: llm is None")
        initialization_failed = True
    if question_generator_llm is None:
        print("Initialization Check Failed: question_generator_llm is None")
        initialization_failed = True
    if source_vectorstore is None:
        print("Initialization Check Failed: source_vectorstore is None")
        initialization_failed = True
    if architecture_qa_vectorstore is None:
        print("Initialization Check Failed: architecture_qa_vectorstore is None")
        initialization_failed = True
    if qa_chain is None:
        print("Initialization Check Failed: qa_chain is None")
        initialization_failed = True

    if initialization_failed:
        print("Aborting Q&A generation due to initialization failure (individual check).")
        return
    # --- End Modified Check ---

    # Original check (commented out)
    # if not all([embedding_model, llm, question_generator_llm, source_vectorstore, architecture_qa_vectorstore, qa_chain]):
    #     print("Aborting Q&A generation due to initialization failure.")
    #     return

    print(f"Generating {num_questions} synthetic architecture-focused Q&A pairs about Hampi...")
    qa_pairs = []

    for i in range(num_questions):
        print(f"\n--- Generating Pair {i+1}/{num_questions} ---")
        # 1. Generate Question
        question = generate_architectural_question(question_generator_llm, topic=topic)
        print(f"Generated Question: {question}")

        if "Error:" in question:
            continue # Skip if question generation failed

        # 2. Generate Answer using RAG
        try:
            print("Generating answer using RAG...")
            # Use invoke instead of deprecated run
            answer_result = qa_chain.invoke({"query": question})
            answer = answer_result.get("result", "Could not retrieve an answer.") # Adjust based on actual output structure
            print(f"Generated Answer: {answer[:150]}...")

            qa_pairs.append({"question": question, "answer": answer})

            # 3. Store in Architecture Vector Store
            qa_text = f"Question: {question}\nAnswer: {answer}"
            topic_keywords = [t for t in hampi_architectural_topics if any(word in question.lower() for word in t.lower().split())]
            metadata_topic = topic_keywords[0] if topic_keywords else "General Hampi architecture"

            architecture_qa_vectorstore.add_texts(
                [qa_text],
                metadatas=[{
                    "question": question,
                    "topic": metadata_topic,
                    "content_type": "architecture"
                }],
                ids=[str(uuid.uuid4())] # Ensure unique IDs
            )
            print(f"Stored Q&A pair in {arch_qa_db_path}")

            # Sleep briefly to avoid potential API rate limits
            time.sleep(1.5)

        except Exception as e:
            print(f"Error processing Q&A pair for question '{question}': {e}")
            time.sleep(1) # Wait a bit longer after an error

    # 4. Save results to TXT file
    save_qa_to_txt(qa_pairs, filename=qa_txt_file)

    # 5. Persist the vector store changes
    try:
        print("Persisting architecture Q&A vector store...")
        architecture_qa_vectorstore.persist() # Ensure data is saved
        print(f"Completed architectural Q&A generation. Stored {len(qa_pairs)} pairs in {arch_qa_db_path}")
    except Exception as e:
        print(f"Error persisting vector store {arch_qa_db_path}: {e}")


def search_architecture_qa_database(query, k=3, arch_qa_db_path=DEFAULT_ARCH_QA_DB_PATH):
    """Searches the architecture-specific Q&A vector store."""
    try:
        print(f"Searching architecture Q&A database at {arch_qa_db_path} for: '{query}'")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = Chroma(persist_directory=arch_qa_db_path, embedding_function=embedding_model)
        results = vector_store.similarity_search(query, k=k)
        print(f"Found {len(results)} results.")
        return results
    except Exception as e:
        print(f"Error searching vector store {arch_qa_db_path}: {e}")
        return []

# Example of how to run directly (optional)
if __name__ == "__main__":
    print("Running store_utils.py directly for testing...")
    # Test Q&A generation
    # process_architectural_qa_batch_and_save(num_questions=2, qa_txt_file="test_hampi_qa.txt", arch_qa_db_path="./test_chroma_arch_qa")

    # Test search (assuming the DB was created and populated)
    # test_query = "Tell me about the pillars in Hampi"
    # search_results = search_architecture_qa_database(test_query, arch_qa_db_path="./test_chroma_arch_qa")
    # if search_results:
    #     for doc in search_results:
    #         print("\n-- Result --")
    #         print(doc.page_content)
    # else:
    #     print("No results found for test query.")
    pass

# --- Add this line for debugging ---
# print(f"DEBUG: GROQ_API_KEY from environment: {os.environ.get('GROQ_API_KEY')}")
# --- End of added line ---

# Existing line where ChatGroq is initialized (This line seems misplaced outside the function, consider removing it if it's redundant)
# llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=os.environ.get("GROQ_API_KEY")) # Or similar