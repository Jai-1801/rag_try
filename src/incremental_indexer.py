# src/incremental_indexer.py
"""
Incremental indexing for large-scale RAG systems (100K+ documents).
Supports batch processing, checkpointing, and progress tracking.
"""

import json
import os
import pickle
from typing import List, Dict, Generator
from datetime import datetime
import hashlib

from chunker import chunk_text
from embedder import embed_texts


class IncrementalIndexBuilder:
    """Build index incrementally with checkpointing."""
    
    def __init__(
        self, 
        checkpoint_dir: str = 'checkpoints',
        batch_size: int = 100,
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ):
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.processed_ids = self._load_processed_ids()
        self.total_chunks = 0
    
    def _load_processed_ids(self) -> set:
        """Load IDs of already processed documents."""
        checkpoint_file = os.path.join(self.checkpoint_dir, 'processed_ids.pkl')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return set()
    
    def _save_processed_ids(self):
        """Save processed document IDs."""
        checkpoint_file = os.path.join(self.checkpoint_dir, 'processed_ids.pkl')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self.processed_ids, f)
    
    def _get_doc_id(self, doc: Dict) -> str:
        """Generate unique ID for document."""
        if 'id' in doc:
            return str(doc['id'])
        # Generate hash-based ID
        content = doc.get('text', '') + doc.get('metadata', {}).get('url', '')
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_documents(
        self, 
        documents: Generator[Dict, None, None],
        indexer
    ):
        """
        Process documents incrementally and add to index.
        
        Expected document format:
        {
            'id': 'unique_id',
            'text': 'document content',
            'metadata': {'title': '', 'url': '', ...}
        }
        """
        batch_texts = []
        batch_metadatas = []
        docs_processed = 0
        docs_skipped = 0
        
        print(f"\n{'='*60}")
        print(f"Starting incremental indexing")
        print(f"Batch size: {self.batch_size}")
        print(f"Already processed: {len(self.processed_ids)} documents")
        print(f"{'='*60}\n")
        
        for doc in documents:
            doc_id = self._get_doc_id(doc)
            
            # Skip already processed documents
            if doc_id in self.processed_ids:
                docs_skipped += 1
                if docs_skipped % 100 == 0:
                    print(f"  Skipped {docs_skipped} already-processed docs")
                continue
            
            # Extract text
            text = doc.get('text', '')
            if not text or not text.strip():
                continue
            
            # Chunk the document
            chunks = chunk_text(text, max_tokens=self.chunk_size, overlap=self.chunk_overlap)
            
            # Prepare chunks for embedding
            for chunk in chunks:
                metadata = {
                    'doc_id': doc_id,
                    'chunk_id': chunk['id'],
                    'text': chunk['text'],
                    'source': doc.get('metadata', {}).get('url', 'unknown'),
                    'title': doc.get('metadata', {}).get('title', ''),
                    'created_at': doc.get('metadata', {}).get('created_at', ''),
                }
                
                batch_texts.append(chunk['text'])
                batch_metadatas.append(metadata)
            
            docs_processed += 1
            self.processed_ids.add(doc_id)
            
            # Process batch when full
            if len(batch_texts) >= self.batch_size:
                self._process_batch(batch_texts, batch_metadatas, indexer)
                batch_texts = []
                batch_metadatas = []
                
                # Save checkpoint every 10 batches
                if docs_processed % (self.batch_size * 10) == 0:
                    self._save_checkpoint(docs_processed)
        
        # Process remaining items
        if batch_texts:
            self._process_batch(batch_texts, batch_metadatas, indexer)
        
        # Final checkpoint
        self._save_checkpoint(docs_processed)
        
        print(f"\n{'='*60}")
        print(f"✓ Indexing complete!")
        print(f"  Documents processed: {docs_processed}")
        print(f"  Documents skipped: {docs_skipped}")
        print(f"  Total chunks indexed: {self.total_chunks}")
        print(f"{'='*60}\n")
    
    def _process_batch(self, texts: List[str], metadatas: List[Dict], indexer):
        """Embed and index a batch of chunks."""
        try:
            print(f"Processing batch of {len(texts)} chunks...")
            
            # Generate embeddings
            vectors = embed_texts(texts)
            
            # Add to index
            indexer.add(vectors, metadatas)
            
            self.total_chunks += len(texts)
            
            print(f"✓ Batch indexed. Total chunks: {self.total_chunks}")
            
        except Exception as e:
            print(f"✗ Error processing batch: {e}")
            raise
    
    def _save_checkpoint(self, docs_processed: int):
        """Save checkpoint."""
        self._save_processed_ids()
        checkpoint_info = {
            'timestamp': datetime.now().isoformat(),
            'docs_processed': docs_processed,
            'total_chunks': self.total_chunks,
            'processed_ids_count': len(self.processed_ids)
        }
        
        checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint_info.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print(f"  💾 Checkpoint saved: {docs_processed} docs, {self.total_chunks} chunks")


def build_index_from_api(api_url: str, api_key: str = None):
    """Complete pipeline: API → Chunks → Embeddings → Index."""
    from api_ingestion import CustomAPIFetcher
    from indexer import FaissIndexer
    
    print("Step 1: Fetching data from API...")
    documents = CustomAPIFetcher.fetch_rest_api(api_url, api_key)
    
    print("\nStep 2: Initializing index...")
    indexer = FaissIndexer(dim=768)  # Gemini embedding dimension
    
    # Try to load existing index
    if indexer.load():
        print("✓ Loaded existing index")
    else:
        print("✓ Starting fresh index")
    
    print("\nStep 3: Building index incrementally...")
    builder = IncrementalIndexBuilder(batch_size=100)
    builder.process_documents(documents, indexer)
    
    print("\nStep 4: Saving final index...")
    indexer.save()
    print("✓ Index saved successfully!")


def build_index_from_jsonl(jsonl_file: str):
    """Build index from JSONL file (one JSON object per line)."""
    from indexer import FaissIndexer
    
    def read_jsonl():
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    
    print(f"Reading from: {jsonl_file}")
    
    indexer = FaissIndexer(dim=768)
    if indexer.load():
        print("✓ Loaded existing index")
    else:
        print("✓ Starting fresh index")
    
    builder = IncrementalIndexBuilder(batch_size=50)
    builder.process_documents(read_jsonl(), indexer)
    
    indexer.save()
    print("✓ Done!")


if __name__ == '__main__':
    # Example: Build from JSONL file
    print("Incremental Index Builder - Example")
    print("="*60)
    
    # Test with sample data
    sample_docs = [
        {
            'id': i,
            'text': f'Sample document {i} with some content about machine learning and AI.',
            'metadata': {'title': f'Doc {i}', 'url': f'http://example.com/{i}'}
        }
        for i in range(10)
    ]
    
    def sample_generator():
        for doc in sample_docs:
            yield doc
    
    from indexer import FaissIndexer
    indexer = FaissIndexer(dim=768)
    
    builder = IncrementalIndexBuilder(batch_size=5)
    builder.process_documents(sample_generator(), indexer)
    
    print("\n✓ Test complete!")