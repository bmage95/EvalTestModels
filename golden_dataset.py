"""
Golden Dataset Loader - Load and process evalset.evalset.json
"""
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCase:
    """Represents a single test case from golden dataset"""
    eval_id: str
    user_input: str
    expected_response: str
    conversation_history: List[Dict] = None
    
    def to_dict(self):
        return {
            'eval_id': self.eval_id,
            'user_input': self.user_input,
            'expected_response': self.expected_response,
            'has_history': bool(self.conversation_history and len(self.conversation_history) > 1)
        }


class GoldenDatasetLoader:
    """Load and manage golden dataset (evalset)"""
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize loader
        
        Args:
            dataset_path: Path to evalset.evalset.json (auto-finds if not provided)
        """
        self.dataset_path = dataset_path or self._find_evalset()
        self.test_cases: List[TestCase] = []
        self.raw_data = None
        
        if self.dataset_path:
            self.load()
    
    def _find_evalset(self) -> Optional[str]:
        """Auto-find evalset.evalset.json"""
        search_paths = [
            Path('my_agent/evalset.evalset.json'),
            Path('evalset.evalset.json'),
            Path('./') / 'my_agent' / 'evalset.evalset.json',
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def load(self) -> bool:
        """Load dataset from JSON file"""
        if not self.dataset_path:
            raise FileNotFoundError("evalset.evalset.json not found")
        
        try:
            with open(self.dataset_path, 'r') as f:
                self.raw_data = json.load(f)
            
            self._parse_test_cases()
            print(f"✓ Loaded {len(self.test_cases)} test cases from {self.dataset_path}")
            return True
        
        except Exception as e:
            print(f"✗ Failed to load dataset: {str(e)}")
            return False
    
    def _parse_test_cases(self):
        """Extract test cases from raw JSON - one per conversation turn with text"""
        self.test_cases = []
        
        if not self.raw_data or 'eval_cases' not in self.raw_data:
            return
        
        for eval_case in self.raw_data['eval_cases']:
            base_eval_id = eval_case.get('eval_id', 'unknown')
            conversation = eval_case.get('conversation', [])
            
            if not conversation:
                continue
            
            # Create a TestCase for EACH turn that has text user_input
            for idx, turn in enumerate(conversation):
                user_input = self._extract_text(turn.get('user_content', {}))
                expected_response = self._extract_text(turn.get('final_response', {}))
                
                # Skip turns without text input (e.g., image-only)
                if not user_input or not expected_response:
                    continue
                
                # Unique ID per turn
                turn_eval_id = f"{base_eval_id}_turn{idx}" if len(conversation) > 1 else base_eval_id
                
                test_case = TestCase(
                    eval_id=turn_eval_id,
                    user_input=user_input,
                    expected_response=expected_response,
                    conversation_history=conversation[:idx+1]  # history up to this turn
                )
                self.test_cases.append(test_case)
    
    def _extract_text(self, content_dict: Dict) -> str:
        """Extract text from parts structure"""
        parts = content_dict.get('parts', [])
        
        texts = []
        for part in parts:
            if isinstance(part, dict) and 'text' in part:
                texts.append(part['text'])
        
        return ' '.join(texts).strip()
    
    def get_test_cases(self) -> List[TestCase]:
        """Get all test cases"""
        return self.test_cases
    
    def get_test_case(self, eval_id: str) -> Optional[TestCase]:
        """Get specific test case by ID"""
        for case in self.test_cases:
            if case.eval_id == eval_id:
                return case
        return None
    
    def get_summary(self) -> Dict:
        """Get dataset summary"""
        return {
            'dataset_path': self.dataset_path,
            'total_cases': len(self.test_cases),
            'eval_set_id': self.raw_data.get('eval_set_id', 'unknown') if self.raw_data else None,
            'eval_set_name': self.raw_data.get('name', 'unknown') if self.raw_data else None,
            'test_cases': [tc.to_dict() for tc in self.test_cases[:5]]  # First 5
        }

    def find_by_input(self, user_text: str) -> Optional[TestCase]:
        """Find a test case whose user_input matches the provided text (case-insensitive, trimmed)."""
        if not user_text:
            return None
        normalized = user_text.strip().lower()
        for tc in self.test_cases:
            if tc.user_input.strip().lower() == normalized:
                return tc
        return None
    
    def export_for_evaluation(self) -> Dict:
        """Export test cases in format ready for evaluation"""
        return {
            'total': len(self.test_cases),
            'cases': [
                {
                    'id': tc.eval_id,
                    'input': tc.user_input,
                    'expected': tc.expected_response
                }
                for tc in self.test_cases
            ]
        }


if __name__ == '__main__':
    # Test the loader
    loader = GoldenDatasetLoader()
    
    print("\n" + "="*60)
    print("GOLDEN DATASET SUMMARY")
    print("="*60)
    print(json.dumps(loader.get_summary(), indent=2))
    
    print("\n" + "="*60)
    print("TEST CASES")
    print("="*60)
    for i, tc in enumerate(loader.get_test_cases()[:3], 1):
        print(f"\n{i}. ID: {tc.eval_id}")
        print(f"   Input: {tc.user_input[:50]}...")
        print(f"   Expected: {tc.expected_response[:50]}...")
