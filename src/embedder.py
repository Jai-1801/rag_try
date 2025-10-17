# src/embedder.py
import os
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

def embed_texts(texts: List[str], model: str = 'models/text-embedding-004', max_retries: int = 3) -> List[List[float]]:
    """Generate embeddings for a list of texts using Google Gemini with retry logic."""
    embeddings = []
    
    # Process one at a time for query (usually just 1 text)
    if len(texts) == 1:
        print(f"Embedding query text...")
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=model,
                    content=texts[0],
                    task_type="retrieval_query"  # Use retrieval_query for questions
                )
                return [result['embedding']]
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise Exception(f"Failed to embed text after {max_retries} attempts: {e}")
    
    # Batch processing for multiple texts (during indexing)
    batch_size = 50
    print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) - 1) // batch_size + 1
        
        print(f"Batch {batch_num}/{total_batches}...", end=' ')
        
        for attempt in range(max_retries):
            try:
                # Process batch one by one (more reliable)
                batch_embeddings = []
                for text in batch:
                    result = genai.embed_content(
                        model=model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    batch_embeddings.append(result['embedding'])
                    time.sleep(0.1)  # Small delay between requests
                
                embeddings.extend(batch_embeddings)
                print("✓")
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}...", end=' ')
                    time.sleep(2 ** attempt)
                else:
                    print(f"✗ Failed: {e}")
                    raise Exception(f"Batch {batch_num} failed after {max_retries} attempts: {e}")
    
    print(f"✓ Embedded {len(embeddings)} chunks successfully")
    return embeddings