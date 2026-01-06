# Vector Database Architecture

## Current Setup

### Components

1. **ChromaDB (Vector Store)**
   - Purpose: Store embeddings of expected answers for semantic retrieval
   - Type: In-memory ephemeral database (resets on restart)
   - Collection: `eval_answers`
   - Location: RAM only (not persisted to disk currently)

2. **Sentence Transformers (Embedding Model)**
   - Model: `all-MiniLM-L6-v2`
   - Dimensions: 384
   - Purpose: Convert text to vector embeddings for semantic similarity
   - Used for: Comparing actual vs expected answer semantics

3. **Semantic Scoring Pipeline**
   ```
   Actual Answer → Embedding Model → Vector (384D)
                                           ↓
                                    Cosine Similarity → Score (0-1)
                                           ↑
   Expected Answer → Embedding Model → Vector (384D)
   ```

## How It Works

### Current Flow
1. When evaluation happens:
   - Actual answer is embedded
   - Expected answer is embedded
   - Cosine similarity is calculated between vectors
   - Score (0-1) is added to evaluation result

### What's NOT Happening Yet
- Expected answers are NOT being stored in ChromaDB
- No semantic retrieval from golden dataset
- Database is ephemeral (lost on restart)

## Proper Setup

### 1. Install Dependencies (Already Done)
```bash
pip install "numpy<2.0"
pip install "chromadb==0.5.3"
pip install "sentence-transformers==2.3.1"
```

### 2. Verify Installation
```python
python -c "import chromadb, sentence_transformers; print('✓ Vector DB ready')"
```

### 3. Enhanced Architecture (What We Should Add)

#### Option A: Persistent Storage
```python
# Store ChromaDB on disk
chroma_client = chromadb.PersistentClient(path="./chroma_db")
```

#### Option B: Pre-populate Golden Dataset Embeddings
```python
# On startup, embed all golden dataset expected answers
for test_case in dataset:
    embedding = embedding_model.encode(test_case.expected_response)
    collection.add(
        ids=[test_case.eval_id],
        embeddings=[embedding.tolist()],
        metadatas=[{"input": test_case.user_input}]
    )
```

#### Option C: Semantic Retrieval
```python
# Find similar expected answers from golden dataset
def find_similar_expected(user_input, top_k=3):
    query_embedding = embedding_model.encode(user_input)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results
```

## Current Issues

### Why ChromaDB Isn't Showing Up
1. **Silent Initialization**: The try-catch suppresses errors
2. **No Persistence**: Data is lost on restart
3. **Not Being Used**: Collection exists but isn't populated or queried
4. **NumPy Compatibility**: Was causing import failures (fixed)

## Recommended Enhancements

### 1. Make ChromaDB Persistent
```python
self.chroma_client = chromadb.PersistentClient(
    path="./vector_db",
    settings=Settings(anonymized_telemetry=False)
)
```

### 2. Pre-Index Golden Dataset
- On startup, load all golden dataset expected answers
- Create embeddings and store in ChromaDB
- Use for semantic search and similarity scoring

### 3. Add Semantic Search Endpoint
- Query: User input → Find similar golden cases
- Response: Top-K similar expected answers with scores

### 4. Hybrid Scoring
- LLM Judge Score (0-1)
- Semantic Similarity (0-1)
- Weighted combination: `final_score = 0.6 * llm_score + 0.4 * semantic_score`

## Testing Setup

### 1. Check Status
```bash
python app.py
# Look for:
# Vector DB: ✓ Available
# Embeddings: ✓ Available
```

### 2. Test Embedding
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Hello world")
print(f"Embedding shape: {embedding.shape}")  # Should be (384,)
```

### 3. Test ChromaDB
```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("test")
collection.add(ids=["1"], embeddings=[[0.1]*384], documents=["test"])
print(f"Count: {collection.count()}")  # Should be 1
```

## What's Working Now

✓ ChromaDB imports successfully
✓ Sentence Transformers loads model
✓ Semantic similarity calculation in evaluation
✓ Scores included in API responses

## What Needs Implementation

- [ ] Persistent ChromaDB storage
- [ ] Pre-index golden dataset on startup
- [ ] Semantic search endpoint
- [ ] Hybrid scoring (LLM + semantic)
- [ ] Vector DB metrics in UI

## Quick Fix: Make It Persistent

Add to `eval_backend.py`:
```python
if CHROMADB_AVAILABLE:
    try:
        # Use persistent storage instead of in-memory
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection("eval_answers")
        print(f"✓ ChromaDB: {self.collection.count()} embeddings loaded")
    except Exception as e:
        print(f"ChromaDB initialization failed: {e}")
        self.chroma_client = None
```

This will:
1. Store data in `./chroma_db` folder
2. Persist across restarts
3. Show count of stored embeddings
