# src/retriever_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import from same directory
from indexer import FaissIndexer
from embedder import embed_texts

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

app = FastAPI()

class QueryIn(BaseModel):
    question: str
    top_k: int = 5

indexer = None

@app.on_event('startup')
async def startup_event():
    """Load the FAISS index on startup."""
    global indexer
    indexer = FaissIndexer(dim=768)  # Gemini embedding dimension
    
    # Change to parent directory to find index files
    os.chdir(parent_dir)
    
    loaded = indexer.load()
    if not loaded:
        raise RuntimeError('Index not found. Run: python src/index_build.py first')
    print(f"✓ Index loaded with {len(indexer.metadatas)} chunks")

@app.post('/query')
async def query(q: QueryIn):
    """Answer a question using retrieved documents."""
    try:
        print(f"\n{'='*60}")
        print(f"Received question: {q.question}")
        print(f"{'='*60}")
        
        # Embed the question with timeout
        print("Step 1: Embedding question...")
        try:
            question_embedding = embed_texts([q.question])[0]
            print("✓ Question embedded successfully")
        except Exception as e:
            print(f"✗ Embedding failed: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to embed question. Check your internet connection and API key. Error: {str(e)}"
            )
        
        # Retrieve similar chunks
        print("Step 2: Searching for similar chunks...")
        results = indexer.query(question_embedding, k=q.top_k)
        print(f"✓ Found {len(results)} relevant chunks")
        
        # Build prompt with retrieved snippets
        system_prompt = """You are an assistant that answers using ONLY the provided sources. 
If the answer is not in the sources, say "I don't know from the provided sources."
Keep answers concise and cite source numbers in brackets like [1], [2]."""
        
        user_prompt = f"Question: {q.question}\n\nSources:\n"
        
        for i, r in enumerate(results):
            meta = r['metadata']
            text = meta.get('text', '')[:500]
            source = meta.get('source', 'unknown')
            user_prompt += f"[{i+1}] (from {source}) {text}\n\n"
        
        user_prompt += "\nAnswer the question using only the information above. Cite sources using [1], [2], etc."
        
        # Call Gemini for answer generation
        print("Step 3: Generating answer with Gemini...")
        try:
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
                request_options={'timeout': 30}  # 30 second timeout
            )
            answer = response.text
            print("✓ Answer generated successfully")
        except Exception as e:
            print(f"✗ Generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate answer. Error: {str(e)}"
            )
        
        # Prepare retrieved sources for response
        retrieved = [
            {
                'source': r['metadata'].get('source'),
                'text': r['metadata'].get('text', '')[:200],
                'score': r['score']
            }
            for r in results
        ]
        
        print(f"{'='*60}\n")
        
        return {'answer': answer, 'retrieved': retrieved}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get('/')
async def root():
    return {
        'message': 'RAG API is running!', 
        'endpoints': {
            'docs': '/docs',
            'query': '/query (POST)'
        }
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)