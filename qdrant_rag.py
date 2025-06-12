import os
import uuid
import json
import pandas as pd
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from concurrent.futures import ThreadPoolExecutor
import openai
from dotenv import load_dotenv
load_dotenv()

# === Configuration ===
collection_name = "csv_data_embeddings"
checkpoint_path = "checkpoint.json"
processed_ids_path = "processed_ids.json"
embedding_batch_size = 128
upsert_workers = 8

# === Global Clients ===
model = None
qdrant_client = None
openai.api_key = os.getenv("OPENAI_API_KEY") # or assign your key directly

def initialize_system(model_name="all-MiniLM-L6-v2"):
    global model, qdrant_client

    # Force CPU mode to avoid CUDA error
    device = "cpu"
    model = SentenceTransformer(model_name, device=device)

    qdrant_client = QdrantClient(
        url="",  # replace
        api_key="",  # replace
        prefer_grpc=False,
        timeout=120
    )

    vector_size = model.get_sentence_embedding_dimension()

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            on_disk_payload=True,
            shard_number=3,
            replication_factor=2
        )
        print(f"✅ Created collection: {collection_name}")
    else:
        print(f"ℹ️ Using existing collection: {collection_name}")

# === Checkpointing ===
def load_checkpoint():
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
            return checkpoint.get("last_index", 0)
    return 0

def load_existing_ids():
    if os.path.exists(processed_ids_path):
        with open(processed_ids_path, "r") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(index):
    with open(checkpoint_path, "w") as f:
        json.dump({"last_index": index}, f)

def save_processed_ids(ids):
    with open(processed_ids_path, "w") as f:
        json.dump(list(ids), f)

# === Embedding & Uploading ===
def batch_embed(texts):
    return model.encode(
        texts,
        batch_size=embedding_batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

def upsert_batch(points):
    try:
        qdrant_client.upsert(collection_name=collection_name, points=points)
    except Exception as e:
        print(f"❌ Upsert failed: {e}")

def process_csv_to_qdrant(csv_path, chunksize=1000):
    start_index = load_checkpoint()
    existing_ids = load_existing_ids()

    total_inserted = 0
    row_offset = start_index
    print(f"\n🚀 Starting batch processing from row {start_index}...\n")

    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunksize, skiprows=range(1, start_index + 1)),
                      desc="Processing", initial=start_index):
        chunk = chunk.reset_index(drop=True)

        if chunk.empty:
            row_offset += chunksize
            continue

        try:
            chunk['text'] = chunk.astype(str).agg(" | ".join, axis=1)
        except Exception as e:
            print(f"❌ Failed to generate 'text' column at row {row_offset}: {e}")
            row_offset += chunksize
            continue

        texts, row_indices = [], []
        for i, row in chunk.iterrows():
            row_index = row_offset + i
            if row_index in existing_ids:
                continue
            texts.append(row['text'])
            row_indices.append(row_index)

        if not texts:
            row_offset += chunksize
            continue

        embeddings = batch_embed(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].cpu().numpy().tolist(),
                payload={"text": texts[i], "row_index": row_indices[i]}
            )
            for i in range(len(texts))
        ]

        batch_size = len(points) // upsert_workers or 1
        batches = [points[i:i + batch_size] for i in range(0, len(points), batch_size)]

        with ThreadPoolExecutor(max_workers=upsert_workers) as executor:
            futures = [executor.submit(upsert_batch, b) for b in batches]
            for f in futures:
                f.result()

        save_checkpoint(row_offset + chunksize)
        existing_ids.update(row_indices)
        save_processed_ids(existing_ids)
        total_inserted += len(points)

        row_offset += chunksize

    print(f"\n✅ Total new vectors inserted: {total_inserted}")
    return total_inserted

# === GPT Prompting ===
def query_gpt(context: str, query: str, model_name="gpt-3.5-turbo"):
    messages = [
        {"role": "system", "content": "You are a strict medical classifier."},
        {"role": "user", "content": f"""
Context:
{context}

Question:
Based on the patient's characteristics, classify the pregnancy risk level in one word and use one of the following: Low, Medium, or High.

Patient characteristics:
{query}

## Instructions ##
1. Output only one word: Low, Medium, or High.
2. Do not explain or repeat the input.
"""}
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI Error: {str(e)}"

# === RAG Query ===
def rag_query(query, top_k=5):
    query_embedding = model.encode([query])[0].tolist()
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        score_threshold=0.2
    )

    if not results:
        return "🤷 No relevant results found."

    context = "\n".join([hit.payload['text'] for hit in results])
    return query_gpt(context=context, query=query)

# === Main Execution ===
if __name__ == "__main__":
    initialize_system()
    process_csv_to_qdrant("filtered_merged_output.csv")

    print("🔍 GPT-based RAG Classifier (type 'quit' to exit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        result = rag_query(user_input)
        print(f"Bot: {result}\n")
