import torch
from transformers import BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # Chat model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = "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"
NF4_CONFIGS=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        )

# generation configs
GEN_TEMPERATURE = 0  # zero for Deterministic results
GEN_SENTENCES = 1
GEN_TOKEN_LIMIT = 1024
GEN_REPEAT_PENALTY = 1.7


# Help data configs
FILES_DIR = "./data"
CHUNKS_SIZE = 250
CHUNKS_OVERLAP_SIZE = 50
DB_NAME = "./local_db"
EMBEDDER_MODEL="sentence-transformers/paraphrase-MiniLM-L12-v2" 