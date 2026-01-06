"""
LiteLLM-powered multi-model evaluator with LLaMA judges and semantic scoring
"""
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except Exception as e:
    print(f"ChromaDB unavailable: {e}")
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception as e:  # catch broader errors (corrupt installs, missing deps)
    print(f"SentenceTransformers unavailable: {e}")
    SENTENCE_TRANSFORMER_AVAILABLE = False


# Gemini model catalog (5 models for comparison)
GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp",
    "gemini-1.0-pro"
]

# Judge models (LLaMA via Ollama with fallback)
PRIMARY_JUDGE = "ollama/llama3"
FALLBACK_JUDGE = "ollama/llama2"


class JudgeType(Enum):
    OLLAMA = "ollama"
    GPT4 = "gpt4"


@dataclass
class EvalScore:
    pass_score: float  # 0-1: Does it meet the requirement?
    threshold_score: float  # 0-1: Quality/closeness to expected
    judge_model: str
    judge_method: str
    reasoning: str
    semantic_score: Optional[float] = None


class JudgeEvaluator:
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_available = bool(self.gemini_api_key)
        self.litellm_available = LITELLM_AVAILABLE
        self.ollama_available = self._check_ollama()
        
        # Vector DB for semantic scoring
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        
        if CHROMADB_AVAILABLE:
            try:
                # Use persistent storage instead of ephemeral in-memory
                self.chroma_client = chromadb.PersistentClient(
                    path="./chroma_db",
                    settings=Settings(anonymized_telemetry=False)
                )
                self.collection = self.chroma_client.get_or_create_collection("eval_answers")
                print(f"✓ ChromaDB initialized: {self.collection.count()} embeddings loaded from disk")
            except Exception as e:
                print(f"✗ ChromaDB initialization failed: {e}")
                self.chroma_client = None
        
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                print("Loading embedding model (all-MiniLM-L6-v2)...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"✓ Embedding model loaded")
            except Exception as e:
                print(f"✗ SentenceTransformer initialization failed: {e}")
                self.embedding_model = None
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running via direct HTTP health check"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            print(f"Ollama health check failed: {e}")
            return False
    
    def _llm_call(self, model: str, messages: List[dict], temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Unified LLM call via LiteLLM orchestration layer"""
        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM not installed. Run: pip install litellm")
        
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"LLM call failed for {model}: {str(e)}")
    
    def _calculate_semantic_similarity(self, actual: str, expected: str) -> float:
        """Calculate semantic similarity using vector embeddings"""
        if not self.embedding_model:
            return 0.5  # Default if embeddings unavailable
        
        try:
            actual_emb = self.embedding_model.encode(actual)
            expected_emb = self.embedding_model.encode(expected)
            
            # Cosine similarity
            import numpy as np
            similarity = np.dot(actual_emb, expected_emb) / (
                np.linalg.norm(actual_emb) * np.linalg.norm(expected_emb)
            )
            return float(max(0.0, min(1.0, similarity)))  # Clamp to [0, 1]
        except Exception as e:
            print(f"Semantic similarity calculation failed: {e}")
            return 0.5
    
    def store_expected_answer(self, eval_id: str, user_input: str, expected_answer: str) -> bool:
        """Store expected answer embedding in vector DB"""
        if not self.collection or not self.embedding_model:
            return False
        
        try:
            # Check if already exists
            existing = self.collection.get(ids=[eval_id])
            if existing['ids']:
                return True  # Already stored
            
            # Generate embedding
            embedding = self.embedding_model.encode(expected_answer)
            
            # Store in ChromaDB
            self.collection.add(
                ids=[eval_id],
                embeddings=[embedding.tolist()],
                documents=[expected_answer],
                metadatas=[{"user_input": user_input}]
            )
            return True
        except Exception as e:
            print(f"Failed to store embedding for {eval_id}: {e}")
            return False
    
    def find_similar_answers(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find similar expected answers from vector DB using semantic search"""
        if not self.collection or not self.embedding_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, self.collection.count())
            )
            
            # Format results
            similar = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    similar.append({
                        'eval_id': doc_id,
                        'expected_answer': results['documents'][0][i],
                        'user_input': results['metadatas'][0][i].get('user_input', ''),
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    })
            return similar
        except Exception as e:
            print(f"Semantic search failed: {e}")
            return []
    
    def evaluate_with_judge(
        self, 
        actual_answer: str, 
        expected_answer: str, 
        judge_model: Optional[str] = None,
        use_semantic: bool = True
    ) -> EvalScore:
        """Evaluate using LLaMA judge (primary + fallback) with semantic scoring"""
        
        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM not installed. Run: pip install litellm")
        
        # Try primary judge, fallback to secondary
        judges_to_try = [judge_model] if judge_model else [PRIMARY_JUDGE, FALLBACK_JUDGE]
        
        prompt = f"""You are an unbiased judge evaluating AI model responses.

ACTUAL ANSWER:
{actual_answer}

EXPECTED ANSWER:
{expected_answer}

Evaluate on two metrics:
1. Pass (0-1): Does the actual answer meet the basic requirement/intent?
2. Threshold (0-1): How close is the actual answer to the expected answer quality-wise?

Respond ONLY in valid JSON on a single line, no code fences, no extra text:
{{"pass_score": <0-1>, "threshold_score": <0-1>, "reasoning": "<brief explanation>"}}"""

        last_error = None
        for judge in judges_to_try:
            try:
                text = self._llm_call(
                    judge,
                    [{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300
                )

                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = text[json_start:json_end]
                    parsed = json.loads(json_str)
                    
                    semantic_score = None
                    reasoning = parsed.get('reasoning', '')
                    
                    if use_semantic:
                        semantic_score = self._calculate_semantic_similarity(actual_answer, expected_answer)
                        reasoning = f"{reasoning} | Semantic: {semantic_score:.2f}"
                
                    return EvalScore(
                        pass_score=float(parsed.get("pass_score", 0.5)),
                        threshold_score=float(parsed.get("threshold_score", 0.5)),
                        judge_model=judge,
                        judge_method="LLM-as-judge + semantic" if use_semantic else "LLM-as-judge",
                        reasoning=reasoning,
                        semantic_score=semantic_score
                    )
            except Exception as e:
                last_error = e
                print(f"Judge {judge} failed: {e}, trying fallback...")
                continue
        
        raise RuntimeError(f"All judge models failed. Last error: {last_error}")
    
    # Backward compatibility
    def evaluate_with_ollama(self, actual_answer: str, expected_answer: str, model: str = "llama3") -> EvalScore:
        """Backward compatibility wrapper for Ollama evaluation"""
        judge = f"ollama/{model}"
        return self.evaluate_with_judge(actual_answer, expected_answer, judge_model=judge)
    
    def evaluate_with_gpt4(self, actual_answer: str, expected_answer: str) -> EvalScore:
        """Backward compatibility wrapper for GPT-4 evaluation (via LiteLLM)"""
        return self.evaluate_with_judge(actual_answer, expected_answer, judge_model="gpt-4o")
    
    def generate_with_gemini(self, prompt: str, model: str = "gemini-1.5-flash") -> str:
        """Generate via Gemini through LiteLLM orchestration"""
        if not self.gemini_api_key and not self.ollama_available:
            raise RuntimeError("Gemini API not configured and no Ollama fallback")
        
        # Set GOOGLE_API_KEY for LiteLLM
        if self.gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = self.gemini_api_key
            # LiteLLM expects the provider prefix for Gemini
            gemini_model = f"gemini/{model}"
        else:
            # Fallback to Ollama
            gemini_model = "ollama/llama3"
        
        try:
            return self._llm_call(
                gemini_model,
                [{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=1000
            )
        except Exception as e:
            # Try Ollama fallback if Gemini fails
            if self.ollama_available and self.gemini_api_key:
                try:
                    return self.generate_with_ollama(prompt)
                except:
                    pass
            raise RuntimeError(f"Generation failed: {str(e)}")
    
    def generate_with_ollama(self, prompt: str, model: str = "llama3") -> str:
        """Generate via Ollama through LiteLLM"""
        return self._llm_call(f"ollama/{model}", [{"role": "user", "content": prompt}], max_tokens=1000)
    
    def generate_with_gpt4(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Generate via GPT-4 through LiteLLM"""
        return self._llm_call(model, [{"role": "user", "content": prompt}], max_tokens=1000)
    
    def evaluate_multi_judge(
        self,
        actual_answer: str,
        expected_answer: str,
        judges: List[JudgeType] = None
    ) -> Dict[str, EvalScore]:
        """Evaluate with multiple judges for consensus"""
        
        if judges is None:
            judges = [JudgeType.OLLAMA]
        
        results = {}
        
        for judge_type in judges:
            try:
                if judge_type == JudgeType.OLLAMA:
                    results[judge_type.value] = self.evaluate_with_judge(
                        actual_answer, expected_answer, judge_model=PRIMARY_JUDGE
                    )
                elif judge_type == JudgeType.GPT4:
                    results[judge_type.value] = self.evaluate_with_gpt4(
                        actual_answer, expected_answer
                    )
            except Exception as e:
                print(f"Warning: {judge_type.value} judge failed: {str(e)}")
                continue
        
        return results
    
    def get_consensus_score(
        self,
        actual_answer: str,
        expected_answer: str,
        judges: List[JudgeType] = None
    ) -> Tuple[float, float, str]:
        """Get average scores from multiple judges"""
        
        results = self.evaluate_multi_judge(actual_answer, expected_answer, judges)
        
        if not results:
            return 0.5, 0.5, "No judges available"
        
        avg_pass = sum(r.pass_score for r in results.values()) / len(results)
        avg_threshold = sum(r.threshold_score for r in results.values()) / len(results)
        
        judge_names = ", ".join(r.judge_model for r in results.values())
        
        return avg_pass, avg_threshold, judge_names


if __name__ == "__main__":
    # Test the evaluator
    evaluator = JudgeEvaluator()
    
    print(f"\n{'='*60}")
    print("EVALUATOR STATUS")
    print(f"{'='*60}")
    print(f"LiteLLM Available: {evaluator.litellm_available}")
    print(f"Ollama Available: {evaluator.ollama_available}")
    print(f"Gemini Available: {evaluator.gemini_available}")
    print(f"ChromaDB Available: {evaluator.chroma_client is not None}")
    print(f"Embeddings Available: {evaluator.embedding_model is not None}")
    print(f"\nGemini Models: {', '.join(GEMINI_MODELS)}")
    print(f"Primary Judge: {PRIMARY_JUDGE}")
    print(f"Fallback Judge: {FALLBACK_JUDGE}")
    print(f"{'='*60}")
    
    if not CHROMADB_AVAILABLE:
        print("\n⚠️  ChromaDB Setup (optional, for semantic scoring):")
        print("   1. pip uninstall -y numpy")
        print("   2. pip install --force-reinstall 'numpy<2.0'")
        print("   3. pip install --force-reinstall 'chromadb==0.5.3'")
        print("   4. pip install --force-reinstall 'sentence-transformers==2.3.1'")
        print("   Then restart the app.\n")
    
    print(f"{'='*60}\n")
