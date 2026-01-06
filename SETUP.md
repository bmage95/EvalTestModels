# 🚀 Model Eval Backend Setup Guide

## Installation & Setup

### Step 1: Install Dependencies

```bash
pip install flask flask-cors openai requests
```

**Note**: Your golden dataset (`my_agent/evalset.evalset.json`) will be auto-loaded!

### Step 2: Setup Ollama (Local, Open-Source Judge)

**Ollama** provides local, unbiased evaluation without vendor lock-in.

#### On Windows:
1. Download from https://ollama.ai
2. Install and run
3. Open PowerShell and pull Llama 2:
   ```bash
   ollama pull llama2
   ```
4. Ollama will run on `http://localhost:11434`

#### On Mac:
```bash
brew install ollama
ollama pull llama2
ollama serve
```

#### On Linux:
```bash
curl https://ollama.ai/install.sh | sh
ollama pull llama2
ollama serve
```

**Verify Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

---

### Step 3: Setup OpenAI API Key (Optional - for GPT-4o as backup judge)

If you want to use GPT-4o as a backup judge (you mentioned you have a free key):

#### OpenAI API Key
```bash
set OPENAI_API_KEY=sk-...  # Windows PowerShell
# or
export OPENAI_API_KEY=sk-...  # Mac/Linux
```

Get key from: https://platform.openai.com/

**Note**: You can use your free tier key, or skip this if using Ollama only.

---

### Step 4: Run Backend Server

From the `JioWorks` folder:

```bash
python app.py
```

You should see:
```
============================================================
MODEL EVAL BACKEND
============================================================
Ollama Status: ✓ Available
Golden Dataset: ✓ Loaded (42 test cases)
API Endpoints:
  GET /api/health - Check backend status
  GET /api/dataset - Get dataset summary
  GET /api/dataset/test-cases - List all test cases
  POST /api/dataset/evaluate-case/<id> - Evaluate single test case
  POST /api/dataset/evaluate-batch - Evaluate multiple test cases
  POST /api/evaluate - Single judge evaluation
  POST /api/evaluate-multi - Multi-judge consensus
  POST /api/evaluate-all-models - Evaluate 3 models

Running on http://localhost:5000
============================================================
```

---

## API Usage

### 1. Check Golden Dataset

**Request:**
```bash
curl http://localhost:5000/api/dataset
```

**Response:**
```json
{
  "available": true,
  "summary": {
    "dataset_path": "my_agent/evalset.evalset.json",
    "total_cases": 42,
    "eval_set_name": "evalset",
    "test_cases": [...]
  }
}
```

### 2. Get All Test Cases

**Request:**
```bash
curl http://localhost:5000/api/dataset/test-cases
```

**Response:**
```json
{
  "total": 42,
  "cases": [
    {
      "eval_id": "casef3945b",
      "user_input": "hi",
      "expected_response": "Hi there! How can I help you today?",
      "has_history": false
    }
  ]
}
```

### 3. Evaluate Single Test Case (Golden Dataset)

**Request:**
```bash
curl -X POST http://localhost:5000/api/dataset/evaluate-case/casef3945b \
  -H "Content-Type: application/json" \
  -d '{
    "actual_answer": "Hello! What do you need?",
    "judge_model": "ollama"
  }'
```

**Response:**
```json
{
  "case_id": "casef3945b",
  "test_input": "hi",
  "expected_response": "Hi there! How can I help you today?",
  "actual_response": "Hello! What do you need?",
  "pass_score": 0.9,
  "threshold_score": 0.85,
  "judge_model": "llama2",
  "reasoning": "Both responses are appropriate greetings..."
}
```

### 4. Evaluate Multiple Test Cases (Batch)

**Request:**
```bash
curl -X POST http://localhost:5000/api/dataset/evaluate-batch \
  -H "Content-Type: application/json" \
  -d '{
    "results": [
      {
        "case_id": "casef3945b",
        "actual_answer": "Hi there!"
      },
      {
        "case_id": "caseXYZ123",
        "actual_answer": "Processing your request..."
      }
    ],
    "judge_model": "ollama"
  }'
```

**Response:**
```json
{
  "total": 2,
  "evaluated": 2,
  "results": [
    {
      "case_id": "casef3945b",
      "pass_score": 0.9,
      "threshold_score": 0.85,
      "judge_model": "llama2"
    }
  ],
  "stats": {
    "avg_pass_score": 0.88,
    "avg_threshold_score": 0.84,
    "passed": 2,
    "failed": 0
  }
}
```

---

### 5. Single Judge Evaluation (Manual)

**Request:**
```bash
curl -X POST http://localhost:5000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "actual_answer": "Paris is the capital of France",
    "expected_answer": "The capital of France is Paris",
    "judge_model": "ollama"
  }'
```

**Response:**
```json
{
  "pass_score": 0.95,
  "threshold_score": 0.92,
  "judge_model": "llama2",
  "judge_method": "LLM-as-judge",
  "reasoning": "The answer correctly identifies Paris as the capital..."
}
```

### 2. Multi-Judge Consensus

**Request:**
```bash
curl -X POST http://localhost:5000/api/evaluate-multi \
  -H "Content-Type: application/json" \
  -d '{
    "actual_answer": "The answer is correct",
    "expected_answer": "Expected response",
    "judges": ["ollama", "claude", "gpt4"]
  }'
```

**Response:**
```json
{
  "individual_results": {
    "ollama": {"pass_score": 0.9, ...},
    "claude": {"pass_score": 0.88, ...},
    "gpt4": {"pass_score": 0.92, ...}
  },
  "consensus": {
    "pass_score": 0.9,
    "threshold_score": 0.89,
    "judges_used": "llama2, Claude 3.5 Sonnet, GPT-4o"
  }
}
```

### 3. Evaluate Multiple Models at Once

**Request:**
```bash
curl -X POST http://localhost:5000/api/evaluate-all-models \
  -H "Content-Type: application/json" \
  -d '{
    "answers": {
      "gemini-1.5-flash": "Gemini Flash response here",
      "gemini-1.5-pro": "Gemini Pro response here"
    },
    "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
    "expected_answer": "Expected answer here",
    "judge_model": "ollama"
  }'
```

---

## Scoring Metrics Explained

### Pass Score (0-1)
- **1.0**: Fully meets requirements/intent ✓
- **0.5**: Partially addresses the question
- **0.0**: Completely off-topic ✗

### Threshold Score (0-1)
- **1.0**: Perfect match to expected answer
- **0.5**: Reasonable quality but differs significantly  
- **0.0**: Very poor quality or completely wrong

### Judge Transparency
- **Judge Model**: Which LLM is evaluating (Llama 2, Claude, GPT-4)
- **Judge Method**: How it evaluates (LLM-as-judge, Rule-based, Hybrid)
- **Reasoning**: The judge's explanation of the scores

---

## Troubleshooting

### "Ollama not available"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama (Windows)
ollama serve

# Start Ollama (Mac/Linux - background)
ollama serve &
```

### "ANTHROPIC_API_KEY not set"
```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your-key-here"

# Mac/Linux
export ANTHROPIC_API_KEY="your-key-here"
```

### Port 5000 already in use
```bash
# Change port in app.py
app.run(debug=True, port=5001)  # Use 5001 instead
```

---

## Architecture

```
┌─────────────┐
│   Frontend  │ (index.html)
│ (HTML/CSS)  │
└──────┬──────┘
       │ HTTP Requests
       ▼
┌──────────────────────┐
│   Flask Backend      │ (app.py - http://localhost:5000)
│  ┌────────────────┐  │
│  │ Judge Router   │  │
│  └────────────────┘  │
└──┬───────┬───────┬──┘
   │       │       │
   ▼       ▼       ▼
┌──────┐ ┌──────┐ ┌──────┐
│Ollama│ │Claude│ │GPT-4 │
│Local │ │ API  │ │ API  │
└──────┘ └──────┘ └──────┘
```

---

## Next Steps

1. ✅ Start Ollama: `ollama serve`
2. ✅ Run backend: `python app.py`
3. ✅ Open frontend: `index.html` in browser
4. ✅ Enter test answers and watch scores populate
5. ✅ (Optional) Add Claude/GPT-4 for consensus

---

## Cost & Performance

| Judge | Cost | Speed | Bias Risk |
|-------|------|-------|-----------|
| **Ollama (Llama 2)** | Free | ~2-5s | Very Low |
| GPT-4o | Free tier / Paid | ~1-2s | Low |

**Recommendation**: Use **Ollama (Llama 2)** as primary judge - completely free and unbiased. Optionally add **GPT-4o** (your free tier) for secondary validation if needed.
