"""
Flask API server to handle evaluation requests from frontend
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from eval_backend import (
    JudgeEvaluator,
    JudgeType,
    GEMINI_MODELS,
    PRIMARY_JUDGE,
    FALLBACK_JUDGE,
)
from golden_dataset import GoldenDatasetLoader
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize evaluator and dataset loader
evaluator = JudgeEvaluator()
dataset_loader = GoldenDatasetLoader()
DEFAULT_GEMINI_MODELS = GEMINI_MODELS  # Use 5 Gemini models from eval_backend

# Pre-index golden dataset in vector DB
def preindex_golden_dataset():
    """Store all golden dataset expected answers in ChromaDB on startup"""
    if not evaluator.collection or not evaluator.embedding_model:
        return
    
    test_cases = dataset_loader.get_test_cases()
    if not test_cases:
        return
    
    print(f"\nIndexing {len(test_cases)} golden dataset cases in vector DB...")
    indexed = 0
    for tc in test_cases:
        if evaluator.store_expected_answer(tc.eval_id, tc.user_input, tc.expected_response):
            indexed += 1
    print(f"✓ Indexed {indexed}/{len(test_cases)} cases in ChromaDB\n")

# Run pre-indexing
preindex_golden_dataset()


@app.route('/api/health', methods=['GET'])
def health():
    """Check if backend is running and Ollama status"""
    return jsonify({
        'status': 'healthy',
        'ollama_available': evaluator.ollama_available,
        'litellm_available': getattr(evaluator, 'litellm_available', False),
        'gemini_available': getattr(evaluator, 'gemini_available', False),
        'gemini_default_models': DEFAULT_GEMINI_MODELS,
        'judge_primary': PRIMARY_JUDGE,
        'judge_fallback': FALLBACK_JUDGE,
        'vector_db_available': evaluator.chroma_client is not None,
        'embeddings_available': evaluator.embedding_model is not None,
        'dataset_loaded': len(dataset_loader.get_test_cases()) > 0,
        'test_cases_available': len(dataset_loader.get_test_cases()),
        'message': 'Backend is running'
    })


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate actual vs expected answer
    
    Expected JSON:
    {
        "actual_answer": "string",
        "expected_answer": "string",
        "judge_model": "string (llama70b/ollama/gpt4)",
        "judges": ["llama70b"] (optional)
    }
    """
    try:
        data = request.json or {}
        
        actual_answer = data.get('actual_answer', '')
        expected_answer = data.get('expected_answer', '')
        judge_model = data.get('judge_model', 'llama70b').lower()
        
        if not actual_answer or not expected_answer:
            return jsonify({
                'error': 'actual_answer and expected_answer are required'
            }), 400
        
        # Single judge evaluation
        if judge_model in ['llama', 'llama70b', 'primary']:
            result = evaluator.evaluate_with_judge(actual_answer, expected_answer)
        elif judge_model in ['llama13b', 'fallback']:
            result = evaluator.evaluate_with_judge(actual_answer, expected_answer, judge_model=FALLBACK_JUDGE)
        elif judge_model == 'ollama':
            if not evaluator.ollama_available:
                return jsonify({
                    'error': 'Ollama not available. Install from https://ollama.ai'
                }), 503
            result = evaluator.evaluate_with_ollama(actual_answer, expected_answer)
        elif judge_model == 'gpt4':
            result = evaluator.evaluate_with_gpt4(actual_answer, expected_answer)
        else:
            return jsonify({'error': f'Unknown judge model: {judge_model}'}), 400
        
        return jsonify({
            'pass_score': result.pass_score,
            'threshold_score': result.threshold_score,
            'judge_model': result.judge_model,
            'judge_method': result.judge_method,
            'reasoning': result.reasoning,
            'semantic_score': result.semantic_score
        })
    
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500


@app.route('/api/evaluate-multi', methods=['POST'])
def evaluate_multi():
    """
    Evaluate with multiple judges for consensus
    
    Expected JSON:
    {
        "actual_answer": "string",
        "expected_answer": "string",
        "judges": ["llama70b", "llama13b", "ollama", "gpt4"] (optional)
    }
    """
    try:
        data = request.json or {}
        
        actual_answer = data.get('actual_answer', '')
        expected_answer = data.get('expected_answer', '')
        judge_names = data.get('judges', ['llama70b'])
        
        if not actual_answer or not expected_answer:
            return jsonify({
                'error': 'actual_answer and expected_answer are required'
            }), 400
        
        formatted = {}
        passes = []
        thresholds = []
        judge_labels = []

        for name in judge_names:
            lname = name.lower()
            try:
                if lname in ['llama', 'llama70b', 'primary']:
                    score = evaluator.evaluate_with_judge(actual_answer, expected_answer)
                elif lname in ['llama13b', 'fallback']:
                    score = evaluator.evaluate_with_judge(actual_answer, expected_answer, judge_model=FALLBACK_JUDGE)
                elif lname == 'ollama':
                    score = evaluator.evaluate_with_ollama(actual_answer, expected_answer)
                elif lname == 'gpt4':
                    score = evaluator.evaluate_with_gpt4(actual_answer, expected_answer)
                else:
                    return jsonify({'error': f'Unknown judge: {name}'}), 400

                formatted[name] = {
                    'pass_score': score.pass_score,
                    'threshold_score': score.threshold_score,
                    'judge_model': score.judge_model,
                    'judge_method': score.judge_method,
                    'reasoning': score.reasoning,
                    'semantic_score': score.semantic_score,
                }
                passes.append(score.pass_score)
                thresholds.append(score.threshold_score)
                judge_labels.append(score.judge_model)
            except Exception as e:
                formatted[name] = {'error': str(e)}

        if not passes:
            return jsonify({'error': 'No judges succeeded. Check LiteLLM/Ollama setup'}), 503

        avg_pass = sum(passes) / len(passes)
        avg_threshold = sum(thresholds) / len(thresholds)

        return jsonify({
            'individual_results': formatted,
            'consensus': {
                'pass_score': avg_pass,
                'threshold_score': avg_threshold,
                'judges_used': ", ".join(judge_labels)
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Multi-judge evaluation failed: {str(e)}'}), 500


@app.route('/api/evaluate-all-models', methods=['POST'])
def evaluate_all_models():
    """
    Evaluate multiple Gemini model responses side-by-side
    
    Expected JSON:
    {
        "answers": {"<model_name>": "string"},
        "expected_answer": "string",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
        "judge_model": "llama70b" (or gpt4/ollama)
    }
    """
    try:
        data = request.json
        
        expected = data.get('expected_answer', '')
        judge_model = data.get('judge_model', 'llama70b').lower()
        answers_payload = data.get('answers') or {}
        if not answers_payload:
            # Backwards compatibility: pick any *_answer fields
            for key, value in (data or {}).items():
                if key.endswith('_answer') and isinstance(value, str):
                    answers_payload[key.replace('_answer', '')] = value

        if not expected:
            return jsonify({'error': 'expected_answer is required'}), 400
        if not answers_payload:
            return jsonify({'error': 'answers payload is required'}), 400

        models = data.get('models') or list(answers_payload.keys())
        results = {}
        
        for model_key in models:
            actual = answers_payload.get(model_key, '')
            if not actual:
                results[model_key] = {'error': 'No answer provided'}
                continue
            
            try:
                if judge_model in ['llama', 'llama70b', 'primary']:
                    score = evaluator.evaluate_with_judge(actual, expected)
                elif judge_model in ['llama13b', 'fallback']:
                    score = evaluator.evaluate_with_judge(actual, expected, judge_model=FALLBACK_JUDGE)
                elif judge_model == 'ollama':
                    if not evaluator.ollama_available:
                        results[model_key] = {'error': 'Ollama not available'}
                        continue
                    score = evaluator.evaluate_with_ollama(actual, expected)
                elif judge_model == 'gpt4':
                    score = evaluator.evaluate_with_gpt4(actual, expected)
                else:
                    results[model_key] = {'error': f'Unknown judge: {judge_model}'}
                    continue

                results[model_key] = {
                    'pass_score': score.pass_score,
                    'threshold_score': score.threshold_score,
                    'judge_model': score.judge_model,
                    'reasoning': score.reasoning,
                    'semantic_score': score.semantic_score
                }
            
            except Exception as e:
                results[model_key] = {'error': str(e)}
        
        return jsonify({
            'results': results,
            'judge_used': judge_model
        })
    
    except Exception as e:
        return jsonify({'error': f'Batch evaluation failed: {str(e)}'}), 500


@app.route('/api/generate-answers', methods=['POST'])
def generate_answers():
    """
    Generate answers from available models to auto-fill the UI.

    Expected JSON:
    {
        "input": "user question",          # optional if eval_id is provided
        "eval_id": "case id",             # optional; will pull user_input from dataset
        "models": ["gemini-1.5-flash","gemini-1.5-pro"] (optional)
    }
    """
    try:
        data = request.json or {}
        eval_id = data.get('eval_id')
        user_input = (data.get('input') or '').strip()
        requested_models = data.get('models', DEFAULT_GEMINI_MODELS)

        # Pull prompt from golden dataset if eval_id is provided
        if eval_id and not user_input:
            tc = dataset_loader.get_test_case(eval_id)
            if tc:
                user_input = tc.user_input
        if not user_input:
            return jsonify({'error': 'input or eval_id with a valid test case is required'}), 400

        prompt = f"Answer the user clearly and concisely.\n\nUser: {user_input}\nAnswer:"
        answers = {}

        for model_name in requested_models:
            try:
                answers[model_name] = {
                    'text': evaluator.generate_with_gemini(prompt, model=model_name)
                }
            except Exception as e:
                answers[model_name] = {'error': str(e)}

        return jsonify({
            'eval_id': eval_id,
            'input_used': user_input,
            'answers': answers
        })

    except Exception as e:
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


@app.route('/api/generate-expected', methods=['POST'])
def generate_expected():
    """
    Generate an expected response when the input is not in the golden dataset.
    Uses Gemini via LiteLLM; falls back to Ollama llama3 if needed.
    Expected JSON:
    {
        "input": "user question"
    }
    """
    try:
        data = request.json or {}
        user_input = (data.get('input') or '').strip()
        if not user_input:
            return jsonify({'error': 'input is required'}), 400

        prompt = f"Provide a concise, high-quality answer for this user message.\n\nUser: {user_input}\nAnswer:"

        try:
            expected = evaluator.generate_with_gemini(prompt, model=DEFAULT_GEMINI_MODELS[0])
            source = f'gemini:{DEFAULT_GEMINI_MODELS[0]}'
        except Exception:
            if evaluator.ollama_available:
                expected = evaluator.generate_with_ollama(prompt, model="llama3")
                source = 'ollama:llama3'
            else:
                return jsonify({'error': 'No generation backend available (Gemini or Ollama)'}), 503

        return jsonify({
            'expected': expected,
            'source': source
        })
    except Exception as e:
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


@app.route('/api/dataset', methods=['GET'])
def get_dataset_info():
    """Get golden dataset summary"""
    return jsonify({
        'available': len(dataset_loader.get_test_cases()) > 0,
        'summary': dataset_loader.get_summary()
    })


@app.route('/api/dataset/test-cases', methods=['GET'])
def get_test_cases():
    """Get all test cases from golden dataset"""
    cases = dataset_loader.get_test_cases()
    return jsonify({
        'total': len(cases),
        'cases': [tc.to_dict() for tc in cases]
    })


@app.route('/api/dataset/find-by-input', methods=['POST'])
def find_by_input():
    """Find a test case by user input (exact, case-insensitive, trimmed)."""
    try:
        data = request.json or {}
        user_input = (data.get('input') or '').strip()
        if not user_input:
            return jsonify({'error': 'input is required'}), 400

        tc = dataset_loader.find_by_input(user_input)
        if not tc:
            return jsonify({'found': False}), 404

        return jsonify({
            'found': True,
            'case': tc.to_dict(),
            'expected': tc.expected_response,
            'input': tc.user_input
        })
    except Exception as e:
        return jsonify({'error': f'Lookup failed: {str(e)}'}), 500


@app.route('/api/dataset/evaluate-case/<case_id>', methods=['POST'])
def evaluate_test_case(case_id):
    """
    Evaluate a single test case from golden dataset
    
    Expected JSON:
    {
        "actual_answer": "string (agent output)",
        "judge_model": "llama70b" (optional)
    }
    """
    try:
        # Get test case from dataset
        test_case = dataset_loader.get_test_case(case_id)
        if not test_case:
            return jsonify({'error': f'Test case {case_id} not found'}), 404
        
        data = request.json
        actual_answer = data.get('actual_answer', '')
        judge_model = data.get('judge_model', 'llama70b').lower()
        
        if not actual_answer:
            return jsonify({'error': 'actual_answer is required'}), 400
        
        # Evaluate using judge
        if judge_model in ['llama', 'llama70b', 'primary']:
            result = evaluator.evaluate_with_judge(actual_answer, test_case.expected_response)
        elif judge_model in ['llama13b', 'fallback']:
            result = evaluator.evaluate_with_judge(actual_answer, test_case.expected_response, judge_model=FALLBACK_JUDGE)
        elif judge_model == 'ollama':
            if not evaluator.ollama_available:
                return jsonify({
                    'error': 'Ollama not available. Install from https://ollama.ai'
                }), 503
            result = evaluator.evaluate_with_ollama(
                actual_answer,
                test_case.expected_response
            )
        elif judge_model == 'gpt4':
            result = evaluator.evaluate_with_gpt4(
                actual_answer,
                test_case.expected_response
            )
        else:
            return jsonify({'error': f'Unknown judge: {judge_model}'}), 400
        
        return jsonify({
            'case_id': case_id,
            'test_input': test_case.user_input,
            'expected_response': test_case.expected_response,
            'actual_response': actual_answer,
            'pass_score': result.pass_score,
            'threshold_score': result.threshold_score,
            'judge_model': result.judge_model,
            'judge_method': result.judge_method,
            'reasoning': result.reasoning,
            'semantic_score': result.semantic_score
        })
    
    except Exception as e:
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500


@app.route('/api/dataset/evaluate-batch', methods=['POST'])
def evaluate_batch():
    """
    Evaluate multiple test cases at once
    
    Expected JSON:
    {
        "results": [
            {
                "case_id": "casef3945b",
                "actual_answer": "agent output 1"
            },
            ...
        ],
        "judge_model": "llama70b"
    }
    """
    try:
        data = request.json
        results_data = data.get('results', [])
        judge_model = data.get('judge_model', 'llama70b').lower()
        
        if not results_data:
            return jsonify({'error': 'results array is required'}), 400
        
        batch_results = {
            'total': len(results_data),
            'evaluated': 0,
            'results': [],
            'stats': {
                'avg_pass_score': 0,
                'avg_threshold_score': 0,
                'passed': 0,
                'failed': 0
            }
        }
        
        pass_scores = []
        threshold_scores = []
        
        for item in results_data:
            case_id = item.get('case_id')
            actual_answer = item.get('actual_answer', '')
            
            if not case_id or not actual_answer:
                batch_results['results'].append({
                    'case_id': case_id,
                    'error': 'Missing case_id or actual_answer'
                })
                batch_results['stats']['failed'] += 1
                continue
            
            test_case = dataset_loader.get_test_case(case_id)
            if not test_case:
                batch_results['results'].append({
                    'case_id': case_id,
                    'error': 'Test case not found'
                })
                batch_results['stats']['failed'] += 1
                continue
            
            try:
                # Evaluate
                if judge_model in ['llama', 'llama70b', 'primary']:
                    result = evaluator.evaluate_with_judge(actual_answer, test_case.expected_response)
                elif judge_model in ['llama13b', 'fallback']:
                    result = evaluator.evaluate_with_judge(actual_answer, test_case.expected_response, judge_model=FALLBACK_JUDGE)
                elif judge_model == 'ollama':
                    if not evaluator.ollama_available:
                        batch_results['results'].append({
                            'case_id': case_id,
                            'error': 'Ollama not available'
                        })
                        batch_results['stats']['failed'] += 1
                        continue
                    result = evaluator.evaluate_with_ollama(
                        actual_answer,
                        test_case.expected_response
                    )
                elif judge_model == 'gpt4':
                    result = evaluator.evaluate_with_gpt4(
                        actual_answer,
                        test_case.expected_response
                    )
                else:
                    batch_results['results'].append({
                        'case_id': case_id,
                        'error': f'Unknown judge: {judge_model}'
                    })
                    batch_results['stats']['failed'] += 1
                    continue
                
                pass_scores.append(result.pass_score)
                threshold_scores.append(result.threshold_score)
                
                batch_results['results'].append({
                    'case_id': case_id,
                    'pass_score': result.pass_score,
                    'threshold_score': result.threshold_score,
                    'judge_model': result.judge_model
                })
                
                batch_results['evaluated'] += 1
                batch_results['stats']['passed'] += 1
            
            except Exception as e:
                batch_results['results'].append({
                    'case_id': case_id,
                    'error': str(e)
                })
                batch_results['stats']['failed'] += 1
        
        # Calculate averages
        if pass_scores:
            batch_results['stats']['avg_pass_score'] = sum(pass_scores) / len(pass_scores)
        if threshold_scores:
            batch_results['stats']['avg_threshold_score'] = sum(threshold_scores) / len(threshold_scores)
        
        return jsonify(batch_results)
    
    except Exception as e:
        return jsonify({'error': f'Batch evaluation failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MODEL EVAL BACKEND (LiteLLM + LLaMA Judges)")
    print("="*60)
    print(f"LiteLLM: {'✓ Available' if evaluator.litellm_available else '✗ Not installed'}")
    print(f"Ollama: {'✓ Running' if evaluator.ollama_available else '✗ Not running'}")
    print(f"Gemini API: {'✓ Configured' if evaluator.gemini_available else '✗ Not configured'}")
    print(f"Vector DB: {'✓ Available' if evaluator.chroma_client else '✗ Not available'}")
    print(f"Embeddings: {'✓ Available' if evaluator.embedding_model else '✗ Not available'}")
    print(f"\nGemini Models: {len(GEMINI_MODELS)} available")
    for i, model in enumerate(GEMINI_MODELS, 1):
        print(f"  {i}. {model}")
    
    # Print dataset info
    dataset_summary = dataset_loader.get_summary()
    if dataset_summary['total_cases'] > 0:
        print(f"Golden Dataset: ✓ Loaded ({dataset_summary['total_cases']} test cases)")
    else:
        print(f"Golden Dataset: ✗ No test cases found")
    
    if not evaluator.chroma_client:
        print("\n⚠️  ChromaDB Setup (optional, for semantic scoring):")
        print("   1. pip uninstall -y numpy")
        print("   2. pip install --force-reinstall 'numpy<2.0'")
        print("   3. pip install --force-reinstall 'chromadb==0.5.3'")
        print("   4. pip install --force-reinstall 'sentence-transformers==2.3.1'")
        print("   Then restart the app.")
    
    print("\nAPI Endpoints:")
    print("  GET /api/health - Check backend status")
    print("  GET /api/dataset - Get dataset summary")
    print("  GET /api/dataset/test-cases - List all test cases")
    print("  POST /api/dataset/find-by-input - Lookup case by user input")
    print("  POST /api/dataset/evaluate-case/<id> - Evaluate single test case")
    print("  POST /api/dataset/evaluate-batch - Evaluate multiple test cases")
    print("  POST /api/evaluate - Single judge evaluation")
    print("  POST /api/evaluate-multi - Multi-judge consensus")
    print("  POST /api/evaluate-all-models - Evaluate multiple Gemini models")
    print("  POST /api/generate-answers - Generate model outputs for UI")
    print("  POST /api/generate-expected - Generate expected when not in golden set")
    print("\nRunning on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
