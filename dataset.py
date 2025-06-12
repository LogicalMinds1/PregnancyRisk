

# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import VectorParams, Distance, PointStruct

# # === Config ===
# CSV_PATH = "filtered_merged_output.csv"  # Replace with your file
# TEXT_COLUMN = "Risk_Level"  # Column to vectorize
# COLLECTION_NAME = "patient_risk_notes"

# QDRANT_URL = "https://53937321-3ec5-4a4f-b96d-48299a9267f0.us-west-2-0.aws.cloud.qdrant.io/"   # Replace with your Qdrant cluster URL
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-GE64pnFobmMIXgG8IcyrhTf6OM9KrFA3phaqgZ1Bo8"                    # Replace with your Qdrant API key


# # === Step 1: Load your dataset ===
# df = pd.read_csv(CSV_PATH)
# df = df.dropna(subset=[TEXT_COLUMN])  # remove rows with missing text

# texts = df[TEXT_COLUMN].tolist()

# # === Step 2: Generate Embeddings ===
# model = SentenceTransformer('all-MiniLM-L6-v2')
# vectors = model.encode(texts)

# # === Step 3: Connect to Qdrant ===
# client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# # === Step 4: Create Collection ===
# client.recreate_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
# )

# # === Step 5: Prepare Payload (metadata) ===
# points = [
#     PointStruct(
#         id=int(i),
#         vector=vectors[i],
#         payload={"text": texts[i], "row_id": int(df.index[i])}
#     )
#     for i in range(len(vectors))
# ]

# # === Step 6: Upload Vectors to Qdrant ===
# client.upsert(collection_name=COLLECTION_NAME, points=points)

# print(f"✅ Uploaded {len(points)} vectors to collection '{COLLECTION_NAME}' in Qdrant.")
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qdrant_cluster_upload")

class QdrantClusterUploader:
    def __init__(self, url, api_key, collection_name):
        """
        Initialize the Qdrant cluster uploader
        
        Args:
            url: Your Qdrant cluster URL (e.g., "https://your-cluster.cloud.qdrant.io:6333")
            api_key: Your Qdrant API key
            collection_name: Name for your collection
        """
        self.config = {
            'url': url,
            'api_key': api_key,
            'collection_name': collection_name,
            'model_name': 'all-MiniLM-L6-v2',
            'text_column': 'Risk_Level',
            'batch_size': 500,  # Optimal for most clusters
            'timeout': 120,
            'shard_number': 3,
            'replication_factor': 2
        }
        
        # Initialize clients and models
        self.client = QdrantClient(
            url=self.config['url'],
            api_key=self.config['api_key'],
            prefer_grpc=True,
            timeout=self.config['timeout']
        )
        self.model = SentenceTransformer(self.config['model_name'])
        
    def _create_collection(self):
        """Ensure collection exists with proper configuration"""
        try:
            self.client.recreate_collection(
                collection_name=self.config['collection_name'],
                vectors_config=models.VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                ),
                shard_number=self.config['shard_number'],
                replication_factor=self.config['replication_factor'],
                on_disk_payload=True  # Recommended for production
            )
            logger.info(f"Collection {self.config['collection_name']} created/updated")
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise

    def _process_batch(self, batch_df, batch_num):
        """Process and upload a single batch"""
        texts = batch_df[self.config['text_column']].tolist()
        vectors = self.model.encode(texts, show_progress_bar=False)
        
        points = [
            models.PointStruct(
                id=int(idx) + (batch_num * self.config['batch_size']),
                vector=vector.tolist(),
                payload={
                    "text": text,
                    "original_id": int(idx),
                    "batch": batch_num
                }
            )
            for idx, text, vector in zip(batch_df.index, texts, vectors)
        ]
        
        # Upload with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client.upsert(
                    collection_name=self.config['collection_name'],
                    points=points,
                    wait=True
                )
                return len(points)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed batch {batch_num} after {max_retries} attempts")
                    raise
                logger.warning(f"Retry {attempt + 1} for batch {batch_num}")

    def upload_data(self, csv_path):
        """
        Process and upload data from CSV to Qdrant cluster
        
        Args:
            csv_path: Path to your CSV file
        """
        logger.info("Starting data upload to Qdrant cluster")
        
        # Step 1: Create collection
        self._create_collection()
        
        # Step 2: Process data in batches
        total_rows = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1  # Count rows
        num_batches = math.ceil(total_rows / self.config['batch_size'])
        
        total_uploaded = 0
        with tqdm(total=num_batches, desc="Uploading batches") as pbar:
            for batch_num, chunk in enumerate(pd.read_csv(
                csv_path,
                chunksize=self.config['batch_size'],
                low_memory=False
            )):
                chunk = chunk.dropna(subset=[self.config['text_column']])
                if not chunk.empty:
                    uploaded = self._process_batch(chunk, batch_num)
                    total_uploaded += uploaded
                pbar.update(1)
                pbar.set_postfix({"vectors": total_uploaded})
        
        logger.info(f"✅ Successfully uploaded {total_uploaded} vectors to {self.config['collection_name']}")
        return total_uploaded

# Example usage
if __name__ == "__main__":
    # Replace these with your actual values
    QDRANT_URL = "https://53937321-3ec5-4a4f-b96d-48299a9267f0.us-west-2-0.aws.cloud.qdrant.io/"   # Replace with your Qdrant cluster URL
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-GE64pnFobmMIXgG8IcyrhTf6OM9KrFA3phaqgZ1Bo8"                    # Replace with your Qdrant API key

    COLLECTION_NAME = "patient_risk_data"
    CSV_PATH = "filtered_merged_output.csv"  # Your CSV file path
    
    # Initialize and run
    uploader = QdrantClusterUploader(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )
    
    # Start the upload process
    uploader.upload_data(CSV_PATH)