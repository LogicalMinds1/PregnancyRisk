from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# === Config ===
QDRANT_URL = "https://53937321-3ec5-4a4f-b96d-48299a9267f0.us-west-2-0.aws.cloud.qdrant.io/"   # Replace with your Qdrant cluster URL
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-GE64pnFobmMIXgG8IcyrhTf6OM9KrFA3phaqgZ1Bo8"                    # Replace with your Qdrant API key
COLLECTION_NAME = "patient_risk_notes"

# === Load the embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Connect to Qdrant ===
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# === Encode the user query ===
query = "Symptoms of preeclampsia"
query_vector = model.encode(query).tolist()

# === Perform semantic search ===
results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=3,
    with_payload=True
)

# === Display results ===
for r in results:
    print(f"Text: {r.payload['text']}\n---")
