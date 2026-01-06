# Model Eval - LiteLLM Architecture

## Overview
Multi-model evaluation system using **LiteLLM** orchestration with **LLaMA judges** and **vector-based semantic scoring**.

## Architecture Components

### 1. Candidate Models (Answer Generators)
**5 Gemini Models** via LiteLLM:
- `gemini-1.5-flash` - Fast, lightweight
- `gemini-1.5-flash-8b` - Even smaller, faster
- `gemini-1.5-pro` - Premium, high quality
- `gemini-2.0-flash-exp` - Experimental, latest features
- `gemini-1.0-pro` - Stable, reliable

### 2. Judge Models (Evaluators)
**LLaMA models** via Ollama:
- **Primary**: `llama3.1:70b` - High accuracy
- **Fallback**: `llama3.1:13b` - Faster, still reliable

Automatic fallback if primary judge fails.

### 3. LiteLLM (Orchestration Layer)
All LLM calls route through LiteLLM:
- ✅ Unified OpenAI-style API
- ✅ Handles Gemini + Ollama seamlessly
- ✅ Automatic retries and timeouts
- ✅ Provider-agnostic code

### 4. Vector Database (Semantic Scoring)
**ChromaDB** + **SentenceTransformers**:
- Computes cosine similarity between actual/expected
- Adds semantic score to LLM judge evaluation
- Uses `all-MiniLM-L6-v2` embedding model

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull LLaMA Models (Ollama)
```bash
# Primary judge
ollama pull llama3.1:70b

# Fallback judge (recommended)
ollama pull llama3.1:13b
```

### 3. Set Gemini API Key
```bash
# Windows
$env:GEMINI_API_KEY="your-api-key-here"

# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"
```

### 4. Run Backend
```bash
python app.py
```

### 5. Open Frontend
Open `index.html` in browser or navigate to `http://localhost:5000`

## Usage

### Compare 2-3 Gemini Models
1. Select Model A (e.g., `gemini-1.5-flash`)
2. Select Model B (e.g., `gemini-1.5-pro`)
3. Optionally select Model C (e.g., `gemini-2.0-flash-exp`)
4. Enter a prompt
5. Click **Send**

### Scoring
Each model gets:
- **Pass Score** (0-1): Intent match
- **Threshold Score** (0-1): Quality vs expected
- **Semantic Score** (0-1): Vector similarity

### Judge Fallback
If `llama3.1:70b` unavailable, automatically falls back to `llama3.1:13b`.

## API Endpoints

### Health Check
```bash
GET /api/health
```

Returns:
- LiteLLM status
- Ollama status
- Gemini API status
- Vector DB status
- Available Gemini models (5)

### Generate Answers
```bash
POST /api/generate-answers
{
  "input": "What is the capital of France?",
  "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
}
```

### Evaluate Multiple Models
```bash
POST /api/evaluate-all-models
{
  "answers": {
    "gemini-1.5-flash": "Paris is the capital...",
    "gemini-1.5-pro": "The capital of France is Paris..."
  },
  "expected_answer": "Paris",
  "judge_model": "ollama"
}
```

## Troubleshooting

### LiteLLM Not Working
```bash
pip install --upgrade litellm
```

### Ollama Not Detected
```bash
# Check if running
ollama list

# Start Ollama
ollama serve
```

### LLaMA Model Not Found
```bash
# Pull missing model
ollama pull llama3.1:70b
ollama pull llama3.1:13b
```

### Gemini API Error
Check your API key:
```bash
# Windows
echo $env:GEMINI_API_KEY

# Linux/Mac
echo $GEMINI_API_KEY
```

Get key at: https://aistudio.google.com/apikey

### Vector DB Issues
```bash
pip install --upgrade chromadb sentence-transformers
```

## Architecture Benefits

1. **No Provider Lock-in**: LiteLLM abstracts providers
2. **Automatic Fallback**: Primary → Fallback judge
3. **Semantic Scoring**: Vector similarity adds objectivity
4. **5 Gemini Models**: Compare across model sizes/versions
5. **Unified API**: All LLM calls through one interface

## File Structure
```
JioWorks/
├── app.py                  # Flask API server
├── eval_backend.py         # LiteLLM + judges + vector DB
├── golden_dataset.py       # Test case loader
├── index.html              # Frontend UI
├── requirements.txt        # Dependencies
└── LITELLM_ARCHITECTURE.md # This file
```

## Next Steps
- Add more embedding models (e.g., BGE, OpenAI)
- Support custom judge prompts
- Add confidence intervals
- Export evaluation results to CSV
