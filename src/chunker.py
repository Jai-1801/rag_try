# src/chunker.py
from typing import List, Dict
import tiktoken

ENCODER = tiktoken.get_encoding('gpt2')

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 100) -> List[Dict]:
    """Split text into overlapping chunks based on token count."""
    
    # Safety check
    if not text or not text.strip():
        print("  ⚠ Warning: Empty or whitespace-only text")
        return []
    
    print(f"  Text length: {len(text)} characters")
    
    try:
        tokens = ENCODER.encode(text)
        print(f"  Token count: {len(tokens)} tokens")
    except Exception as e:
        print(f"  ❌ Error encoding text: {e}")
        return []
    
    if len(tokens) == 0:
        print("  ⚠ Warning: No tokens generated")
        return []
    
    start = 0
    chunks = []
    idx = 0
    
    # Safety limit: max 1000 chunks per document
    max_chunks = 1000
    
    while start < len(tokens) and idx < max_chunks:
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        
        try:
            chunk_text = ENCODER.decode(chunk_tokens)
        except Exception as e:
            print(f"  ❌ Error decoding chunk {idx}: {e}")
            start = end - overlap
            continue
        
        chunks.append({
            'id': f'chunk_{idx}',
            'text': chunk_text,
            'start_token': start,
            'end_token': end,
        })
        idx += 1
        start = end - overlap
        
        # Progress indicator for large files
        if idx % 10 == 0:
            print(f"    ... created {idx} chunks so far")
    
    print(f"  ✓ Created {len(chunks)} chunks total")
    return chunks