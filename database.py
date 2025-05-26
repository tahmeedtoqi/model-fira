
"""this code is used for processing the QA dataset from standford university into FIASS index and metadata file.
which was later used in the training code"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ----- Step 1: Load Your JSON Dataset -----
with open("train-v1.1.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# ----- Step 2: Initialize the Sentence Transformer -----
# You can change the model name if needed.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----- Step 3: Extract QA Pairs and Prepare Text for Embedding -----
# We will create one record per QA pair.
records = []  # to store metadata for each record
texts_to_embed = []  # to store the text that will be embedded

for article in dataset["data"]:
    # (Optional) Use the title if needed: article["title"]
    for paragraph in article.get("paragraphs", []):
        context = paragraph.get("context", "")
        for qa in paragraph.get("qas", []):
            question = qa.get("question", "")
            # We take the first answer in the list.
            answer = qa.get("answers", [{}])[0].get("text", "")
            # Create a combined text. You might experiment with different formulations.
            combined_text = f"Question: {question}\nContext: {context}"
            
            # Save the record metadata and the text used for embedding.
            records.append({
                "question": question,
                "context": context,
                "answer": answer,
                "id": qa.get("id", "")
            })
            texts_to_embed.append(combined_text)

# ----- Step 4: Compute Embeddings for Each Record -----
# This creates a list of vector embeddings for our texts.
embeddings = embedder.encode(texts_to_embed)
embeddings = np.array(embeddings, dtype=np.float16)

# ----- Step 5: Create and Populate the FAISS Index -----
embedding_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)  # Using L2 distance metric
index.add(embeddings)  # Add all embeddings to the index

# Save the FAISS index to a file
faiss.write_index(index, "qa_dataset.index")
print("FAISS index saved as 'qa_dataset.index'.")

# ----- Step 6: Save the Metadata for Retrieval -----
# Save the records metadata so you can later look up the original QA pair.
with open("qa_metadata.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)
print("QA metadata saved as 'qa_metadata.json'.")


# Load FAISS index and metadata
index = faiss.read_index("qa_dataset.index")
with open("qa_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Function to search for the best matching context
def search_faiss(question):
    query_embedding = embedder.encode([question]).astype(np.float16)
    D, I = index.search(query_embedding, k=1)  # Retrieve top 1 match
    best_match_idx = I[0][0]
    
    if best_match_idx != -1:
        matched_data = metadata[best_match_idx]
        return matched_data["context"], matched_data["answer"]
    else:
        return "No relevant context found.", "No answer available."

# Example query
question = "Which NFL team represented the AFC at Super Bowl 50?"
retrieved_context, answer = search_faiss(question)

print(f"Best Matching Context: {retrieved_context}")
print(f"Predicted Answer: {answer}")
