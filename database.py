import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

class DatabaseCreator:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def create_for_channel(self, channel_id):
        """Create FAISS database for a specific channel"""
        creator_dir = f"creators/{channel_id}"
        
        # Check if directory exists
        if not os.path.exists(creator_dir):
            raise ValueError(f"No directory found for channel {channel_id}")
        
        # Load transcripts
        csv_path = f"{creator_dir}/transcripts.csv"
        if not os.path.exists(csv_path):
            raise ValueError(f"No transcripts found at {csv_path}")
            
        print("📚 Loading transcripts...")
        df = pd.read_csv(csv_path)
        
        if df.empty:
            raise ValueError("Transcript file is empty!")
        
        texts = df["Transcript"].tolist()
        
        # Create embeddings
        print("🔄 Converting transcripts to AI-friendly format...")
        embeddings = []
        for text in texts:
            if isinstance(text, str):  # Ensure text is valid
                embedding = self.model.encode(text)
                embeddings.append(embedding)
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Create and save FAISS index
        print("💾 Creating searchable database...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save files
        faiss_path = f"{creator_dir}/faiss_index"
        texts_path = f"{creator_dir}/texts.npy"
        
        faiss.write_index(index, faiss_path)
        np.save(texts_path, texts)
        
        print(f"✅ Successfully created database files in {creator_dir}")
        return faiss_path, texts_path

# Only run if script is run directly
if __name__ == "__main__":
    channel_id = input("Enter YouTube channel ID: ")
    creator = DatabaseCreator()
    creator.create_for_channel(channel_id)