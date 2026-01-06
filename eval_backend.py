"""
Ollama local + GPT-4 multi-model judge evaluator
"""
import os
import json
import requests
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import openai
except ImportError:
    openai = None


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


class JudgeEvaluator:
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.ollama_available = self._check_ollama()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_available = bool(self.gemini_api_key)
    
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def evaluate_with_ollama(
        self, 
        actual_answer: str, 
        expected_answer: str, 
        model: str = "llama3"
    ) -> EvalScore:
        """Evaluate using local Ollama model (open source, unbiased)"""
        
        if not self.ollama_available:
            raise RuntimeError(
                "Ollama not running. Install from https://ollama.ai and run: ollama serve"
            )
        
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

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            text = (result.get("response", "") or "").strip()

            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = text[json_start:json_end]
                try:
                    parsed = json.loads(json_str)
                except Exception as parse_err:
                    raise RuntimeError(
                        f"Ollama evaluation failed: could not parse JSON: {parse_err} | raw='{text[:200]}'"
                    ) from parse_err
                
                return EvalScore(
                    pass_score=float(parsed.get("pass_score", 0.5)),
                    threshold_score=float(parsed.get("threshold_score", 0.5)),
                    judge_model=model,
                    judge_method="LLM-as-judge",
                    reasoning=parsed.get("reasoning", "")
                )
            raise RuntimeError(f"Ollama evaluation failed: no JSON object in response | raw='{text[:200]}'")
        except Exception as e:
            raise RuntimeError(f"Ollama evaluation failed: {str(e)}")
    
    def evaluate_with_gpt4(
        self, 
        actual_answer: str, 
        expected_answer: str
    ) -> EvalScore:
        """Evaluate using GPT-4 API"""
        
        if not self.openai_client:
            raise RuntimeError(
                "OpenAI API not configured. Set OPENAI_API_KEY environment variable"
            )
        
        prompt = f"""You are an unbiased judge evaluating AI model responses.

ACTUAL ANSWER:
{actual_answer}

EXPECTED ANSWER:
{expected_answer}

Evaluate on two metrics:
1. Pass (0-1): Does the actual answer meet the basic requirement/intent? 
2. Threshold (0-1): How close is the actual answer to the expected answer quality-wise?

Respond in JSON format:
{{"pass_score": <0-1>, "threshold_score": <0-1>, "reasoning": "<brief explanation>"}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            text = response.choices[0].message.content
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = text[json_start:json_end]
                parsed = json.loads(json_str)
                
                return EvalScore(
                    pass_score=float(parsed.get("pass_score", 0.5)),
                    threshold_score=float(parsed.get("threshold_score", 0.5)),
                    judge_model="GPT-4o",
                    judge_method="LLM-as-judge",
                    reasoning=parsed.get("reasoning", "")
                )
        except Exception as e:
            raise RuntimeError(f"GPT-4 evaluation failed: {str(e)}")

    def generate_with_ollama(self, prompt: str, model: str = "llama3") -> str:
        """Generate a response using a local Ollama model"""
        if not self.ollama_available:
            raise RuntimeError(
                "Ollama not running. Install from https://ollama.ai and run: ollama serve"
            )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as http_err:
                if response.status_code == 404:
                    raise RuntimeError(
                        f"Ollama model '{model}' not found. Pull it with: ollama pull {model}"
                    ) from http_err
                raise
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")

    def generate_with_gemini(self, prompt: str, model: str = "gemini-1.5-flash") -> str:
        """Generate a response using Gemini API; falls back to Ollama when not configured."""
        if self.gemini_api_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.6}
            }
            try:
                response = requests.post(
                    url,
                    params={"key": self.gemini_api_key},
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                data = response.json() or {}
                candidates = data.get("candidates") or []
                if not candidates:
                    raise RuntimeError("no candidates returned")
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    text = part.get("text")
                    if text:
                        return text.strip()
                raise RuntimeError("empty response from Gemini")
            except Exception as e:
                raise RuntimeError(f"Gemini generation failed: {str(e)}")

        if self.ollama_available:
            return self.generate_with_ollama(prompt, model="llama3")

        raise RuntimeError("Gemini API not configured and no Ollama fallback available")

    def generate_with_gpt4(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Generate a response using GPT-4o"""
        if not self.openai_client:
            raise RuntimeError(
                "OpenAI API not configured. Set OPENAI_API_KEY environment variable"
            )

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a concise, helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"GPT-4 generation failed: {str(e)}")
    
    def evaluate_multi_judge(
        self,
        actual_answer: str,
        expected_answer: str,
        judges: List[JudgeType] = None
    ) -> Dict[str, EvalScore]:
        """Evaluate with multiple judges for consensus"""
        
        if judges is None:
            judges = [JudgeType.OLLAMA]  # Default to Ollama (unbiased)
        
        results = {}
        
        for judge_type in judges:
            try:
                if judge_type == JudgeType.OLLAMA:
                    results[judge_type.value] = self.evaluate_with_ollama(
                        actual_answer, expected_answer, model="llama3"
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
    
    actual = "The capital of France is Paris."
    expected = "Paris is the capital and most populous city of France."
    
    print(f"Ollama available: {evaluator.ollama_available}")
    
    if evaluator.ollama_available:
        result = evaluator.evaluate_with_ollama(actual, expected)
        print(f"\nOllama Evaluation:")
        print(f"  Pass Score: {result.pass_score}")
        print(f"  Threshold Score: {result.threshold_score}")
        print(f"  Reasoning: {result.reasoning}")
