# src/index_build.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import list_text_files, read_text
from src.chunker import chunk_text
from src.embedder import embed_texts
from src.indexer import FaissIndexer

DATA_DIR = 'data'

def build_index():
    """Build FAISS index from text files in data directory."""
    print("=" * 50)
    print("Starting RAG Index Building Process")
    print("=" * 50)
    
    files = list_text_files(DATA_DIR)
    
    if not files:
        print(f"❌ No .txt files found in {DATA_DIR}")
        print(f"Please add some .txt files to the '{DATA_DIR}' folder first.")
        return
    
    print(f"\n✓ Found {len(files)} text file(s)")
    for f in files:
        print(f"  - {f}")
    
    print(f"\n[1/4] Chunking documents...")
    all_chunks = []
    metadatas = []
    
    for idx, fpath in enumerate(files, 1):
        print(f"\n  Processing file {idx}/{len(files)}: {os.path.basename(fpath)}")
        
        try:
            txt = read_text(fpath)
            
            if not txt.strip():
                print(f"  ⚠ Skipping empty file: {fpath}")
                continue
            
            print(f"  File size: {len(txt)} characters")
            chunks = chunk_text(txt, max_tokens=300, overlap=50)  # Smaller chunks for speed
            
            if not chunks:
                print(f"  ⚠ No chunks created from {fpath}")
                continue
            
            for c in chunks:
                meta = {
                    'source': fpath,
                    'chunk_id': c['id'],
                    'text': c['text']
                }
                all_chunks.append(c['text'])
                metadatas.append(meta)
                
        except Exception as e:
            print(f"  ❌ Error processing {fpath}: {e}")
            continue
    
    if not all_chunks:
        print(f"\n❌ No chunks created from any files!")
        return
    
    print(f"\n✓ Total chunks created: {len(all_chunks)}")
    
    print(f"\n[2/4] Generating embeddings (this may take 1-2 minutes)...")
    print(f"  Note: First time may be slower due to API initialization")
    
    try:
        vectors = embed_texts(all_chunks)
    except Exception as e:
        print(f"\n❌ Error generating embeddings: {e}")
        print("\nPossible issues:")
        print("  1. Check your GOOGLE_API_KEY in .env file")
        print("  2. Verify API key at: https://aistudio.google.com/app/apikey")
        print("  3. Check internet connection")
        return
    
    if not vectors or len(vectors) != len(all_chunks):
        print(f"❌ Embedding failed. Expected {len(all_chunks)}, got {len(vectors)}")
        return
    
    print(f"\n[3/4] Building FAISS index...")
    dim = len(vectors[0])
    print(f"  Embedding dimension: {dim}")
    
    indexer = FaissIndexer(dim)
    indexer.add(vectors, metadatas)
    
    print(f"\n[4/4] Saving index to disk...")
    indexer.save()
    
    print("\n" + "=" * 50)
    print(f"✓ SUCCESS! Index built with {len(vectors)} chunks")
    print("=" * 50)
    print(f"\nFiles created:")
    print(f"  - faiss.index")
    print(f"  - faiss_meta.pkl")
    print(f"\nNext step: Run the API server")
    print(f"  python src/retriever_api.py")

if __name__ == '__main__':
    try:
        build_index()
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()