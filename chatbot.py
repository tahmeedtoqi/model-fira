"""
Sample from a trained model
"""
from contextlib import nullcontext
import torch
import tiktoken
import math
import inspect
from dataclasses import dataclass
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from Fira import FIRA, FIRAConfig
from Fira import FIRAConfig
import torch
import torch.nn as nn
from torch.nn import functional as F


enc = tiktoken.get_encoding("gpt2")


model_path = "ckpt.pt"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model config and set model parameters
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

# Load the checkpoint
checkpoint = torch.load(model_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
conf = FIRAConfig(**model_args)
model = FIRA(conf)
state_dict = checkpoint['model']  
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print("Model loaded successfully!")


faiss_index_bytes = checkpoint.get('faiss_index', None)
faiss_metadata = checkpoint.get('faiss_metadata', None)
if faiss_index_bytes is not None:
    faiss_index = faiss.deserialize_index(faiss_index_bytes)
    print("Retrieval FAISS index loaded from checkpoint.")
else:
    faiss_index = None
    print("No FAISS index found in checkpoint.")
if faiss_metadata is None:
    faiss_metadata = []
    print("No retrieval metadata found in checkpoint.")


retrieval_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print("Retrieval model loaded.")


def retrieve_answer(user_prompt):
    """
    Given a user prompt, compute its embedding (prepended with "Question:" to mirror training),
    search the FAISS index, and check if the retrieved QA pair's question contains (or is contained in)
    the user prompt (case-insensitive). If yes, return the stored context and answer.
    Otherwise, return (None, None).
    """
    if faiss_index is None or len(faiss_metadata) == 0:
        return None, None

    # Format the query to resemble the stored combined text.
    query_text = f"Question: {user_prompt}"
    # Compute embedding (ensure float32, same as used during training)
    query_emb = retrieval_model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    # Search FAISS index (k=1)
    D, I = faiss_index.search(query_emb, k=1)
    best_idx = I[0][0]
    retrieved_record = faiss_metadata[best_idx]
    retrieved_question = retrieved_record.get("question", "").lower()
    user_lower = user_prompt.lower()
    if user_lower in retrieved_question or retrieved_question in user_lower:
        return retrieved_record.get("context", ""), retrieved_record.get("answer", "")
    else:
        return None, None

# ========= Text Generation Function ==========
def chat_with_model(model, user_input, max_length=50, top_p=0.95):
    """
    Generate text using the model if no retrieval match is found.
    """
    input_tokens = enc.encode(user_input)
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)  # Fixed seed for reproducibility

    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            while input_tensor.size(1) < max_length:
                logits, _ = model(input_tensor)
                logits = logits[:, -1, :] / 1.3  # temperature=1.0
                # Top-p sampling
                sorted_probs, sorted_indices = torch.sort(F.softmax(logits, dim=-1), descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs > top_p
                sorted_probs[mask] = 0
                if sorted_probs.sum(dim=-1, keepdim=True).item() == 0:
                    sorted_probs = torch.ones_like(sorted_probs) / sorted_probs.size(-1)
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                ix = torch.multinomial(sorted_probs, 1, generator=sample_rng)
                xcol = torch.gather(sorted_indices, -1, ix)
                input_tensor = torch.cat((input_tensor, xcol), dim=1)
    output_tokens = input_tensor.squeeze().tolist()
    # Decode only the generated tokens (after the original input)
    response = enc.decode(output_tokens[len(input_tokens):])

    return response.strip()

# ========= Chat Loop ==========
print("You can now chat with your model! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # First, try retrieval-based answer
    context_ret, answer_ret = retrieve_answer(user_input)
    if context_ret is not None and answer_ret is not None:
        print("FIRA (retrieval):")
        print(f"Answer: {answer_ret}")
        print(f"Context: {context_ret}")
        
    else:
        response = chat_with_model(model, user_input)
        print("FIRA (generation):", response)
