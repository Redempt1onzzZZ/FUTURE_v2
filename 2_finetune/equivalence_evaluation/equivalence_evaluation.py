import os
import json
import numpy as np
import warnings
import traceback
from typing import List, Dict, Any, Tuple, Union
import argparse
import contextlib
import io
import sys

# Disable warnings to avoid output interference
warnings.filterwarnings('ignore')

class EquivalenceEvaluator:
    def __init__(self, input_dir: str, output_dir: str, threshold: float = 0.01):
        """
        Initialize the equivalence evaluator
        
        Args:
            input_dir: Directory containing mutated code pairs
            output_dir: Directory to save equivalent code pairs
            threshold: Distance threshold for equivalence (default: 0.01)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.threshold = threshold
        
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics
        self.total_pairs = 0
        self.equivalent_pairs = 0
        self.error_pairs = 0
    
    def safe_execute_code(self, code: str) -> Tuple[bool, Any]:
        """
        Safely execute code and return the result
        
        Args:
            code: Python code to execute
            
        Returns:
            Tuple of (success, result)
        """
        try:
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': __builtins__,
                'numpy': np,
                'np': np,
                'math': __import__('math'),
                'random': __import__('random'),
                'float': float,
                'int': int,
                'list': list,
                'dict': dict,
                'len': len,
                'range': range,
                'abs': abs,
                'max': max,
                'min': min,
                'sum': sum,
                'any': any,
                'all': all,
            }
            
            # Try to import common ML libraries
            try:
                import torch
                safe_globals['torch'] = torch
            except ImportError:
                pass
            
            try:
                import tensorflow as tf
                safe_globals['tf'] = tf
                safe_globals['tensorflow'] = tf
            except ImportError:
                pass
                
            try:
                import mlx.core as mx
                safe_globals['mx'] = mx
            except ImportError:
                pass
            
            # Capture stdout to prevent printing
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                # Execute the code
                local_vars = {}
                exec(code, safe_globals, local_vars)
                
                # Find the result (usually the last expression or assigned variable)
                result = None
                if local_vars:
                    # Try to find result variable or take the last assigned value
                    if 'result' in local_vars:
                        result = local_vars['result']
                    elif 'output' in local_vars:
                        result = local_vars['output']
                    else:
                        # Take the last non-internal variable
                        for key, value in local_vars.items():
                            if not key.startswith('_'):
                                result = value
                
                return True, result
                
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                
        except Exception as e:
            return False, str(e)
    
    def normalize_result(self, result: Any) -> np.ndarray:
        """
        Normalize result to numpy array for distance calculation
        
        Args:
            result: Result from code execution
            
        Returns:
            Normalized numpy array
        """
        if result is None:
            return np.array([0.0])
        
        # Handle different types
        if isinstance(result, (int, float)):
            return np.array([float(result)])
        elif isinstance(result, complex):
            return np.array([result.real, result.imag])
        elif isinstance(result, (list, tuple)):
            try:
                return np.array(result, dtype=float).flatten()
            except (ValueError, TypeError):
                # If conversion fails, convert to string lengths
                return np.array([len(str(item)) for item in result], dtype=float)
        elif isinstance(result, np.ndarray):
            return result.flatten().astype(float)
        elif hasattr(result, 'numpy'):  # PyTorch tensor
            try:
                return result.detach().cpu().numpy().flatten().astype(float)
            except:
                return np.array([0.0])
        elif hasattr(result, 'eval'):  # TensorFlow tensor
            try:
                import tensorflow as tf
                if tf.executing_eagerly():
                    return result.numpy().flatten().astype(float)
                else:
                    with tf.Session() as sess:
                        return sess.run(result).flatten().astype(float)
            except:
                return np.array([0.0])
        else:
            # Convert to string and use length
            return np.array([len(str(result))], dtype=float)
    
    def calculate_euclidean_distance(self, result1: Any, result2: Any) -> float:
        """
        Calculate Euclidean distance between two results
        
        Args:
            result1: First result
            result2: Second result
            
        Returns:
            Euclidean distance
        """
        try:
            # Normalize both results
            arr1 = self.normalize_result(result1)
            arr2 = self.normalize_result(result2)
            
            # Handle NaN and Inf values
            arr1 = np.nan_to_num(arr1, nan=0.0, posinf=1e10, neginf=-1e10)
            arr2 = np.nan_to_num(arr2, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Make arrays same length by padding with zeros
            max_len = max(len(arr1), len(arr2))
            if len(arr1) < max_len:
                arr1 = np.pad(arr1, (0, max_len - len(arr1)), 'constant')
            if len(arr2) < max_len:
                arr2 = np.pad(arr2, (0, max_len - len(arr2)), 'constant')
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(arr1 - arr2)
            
            return float(distance)
            
        except Exception as e:
            # If distance calculation fails, assume they're different
            return float('inf')
    
    def evaluate_code_pair(self, code_pair: Dict[str, str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if a code pair is equivalent
        
        Args:
            code_pair: Dictionary containing code implementations
            
        Returns:
            Tuple of (is_equivalent, debug_info)
        """
        debug_info = {
            'results': {},
            'distances': {},
            'errors': {}
        }
        
        # Get all available implementations
        implementations = []
        for impl_type in ['original', 'pytorch', 'tensorflow']:
            if impl_type in code_pair and code_pair[impl_type].strip():
                implementations.append(impl_type)
        
        if len(implementations) < 2:
            return False, {'error': 'Less than 2 implementations available'}
        
        # Execute all implementations
        results = {}
        for impl_type in implementations:
            success, result = self.safe_execute_code(code_pair[impl_type])
            if success:
                results[impl_type] = result
                debug_info['results'][impl_type] = str(result)[:100]  # First 100 chars
            else:
                debug_info['errors'][impl_type] = str(result)[:200]  # Error message
        
        # Need at least 2 successful executions
        if len(results) < 2:
            return False, debug_info
        
        # Calculate pairwise distances
        impl_list = list(results.keys())
        distances = []
        
        for i in range(len(impl_list)):
            for j in range(i + 1, len(impl_list)):
                impl1, impl2 = impl_list[i], impl_list[j]
                distance = self.calculate_euclidean_distance(results[impl1], results[impl2])
                distances.append(distance)
                debug_info['distances'][f'{impl1}-{impl2}'] = distance
        
        # Check if all distances are below threshold
        max_distance = max(distances) if distances else float('inf')
        is_equivalent = max_distance <= self.threshold
        
        debug_info['max_distance'] = max_distance
        debug_info['is_equivalent'] = is_equivalent
        
        return is_equivalent, debug_info
    
    def process_file(self, input_file: str, lib_name: str):
        """
        Process a single mutated code pair file
        
        Args:
            input_file: Path to the input JSON file
            lib_name: Library name (for output directory structure)
        """
        # Create output directory
        lib_output_dir = os.path.join(self.output_dir, lib_name)
        os.makedirs(lib_output_dir, exist_ok=True)
        
        # Load mutated code pairs
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                code_pairs = json.load(f)
        except Exception as e:
            print(f"Error loading {input_file}: {e}")
            return
        
        if not code_pairs:
            print(f"No code pairs found in {input_file}, skipping")
            return
        
        # Get API name from file name
        api_name = os.path.basename(input_file).replace('_mutated.json', '')
        
        # Evaluate each code pair
        equivalent_pairs = []
        debug_logs = []
        
        print(f"Evaluating {len(code_pairs)} code pairs for {api_name}...")
        
        for i, code_pair in enumerate(code_pairs):
            self.total_pairs += 1
            
            try:
                is_equivalent, debug_info = self.evaluate_code_pair(code_pair)
                
                if is_equivalent:
                    equivalent_pairs.append(code_pair)
                    self.equivalent_pairs += 1
                else:
                    self.error_pairs += 1
                
                # Log debug info for first few pairs or problematic ones
                if i < 5 or not is_equivalent:
                    debug_logs.append({
                        'pair_index': i,
                        'is_equivalent': is_equivalent,
                        'debug_info': debug_info
                    })
                    
            except Exception as e:
                self.error_pairs += 1
                debug_logs.append({
                    'pair_index': i,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        
        # Save equivalent code pairs
        if equivalent_pairs:
            output_file = os.path.join(lib_output_dir, f"{api_name}_equivalent.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(equivalent_pairs, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(equivalent_pairs)}/{len(code_pairs)} equivalent code pairs for {api_name}")
        else:
            print(f"No equivalent code pairs found for {api_name}")
        
        # Save debug logs
        debug_file = os.path.join(lib_output_dir, f"{api_name}_debug.json")
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_logs, f, indent=2, ensure_ascii=False)
    
    def process_all_files(self):
        """Process all mutated code pair files in the input directory"""
        print(f"Starting equivalence evaluation with distance threshold: {self.threshold}")
        print("=" * 50)
        
        for lib_dir in os.listdir(self.input_dir):
            lib_path = os.path.join(self.input_dir, lib_dir)
            
            if not os.path.isdir(lib_path):
                continue
            
            print(f"\nProcessing {lib_dir} library...")
            
            # Process each JSON file in the library directory
            for file_name in os.listdir(lib_path):
                if file_name.endswith('_mutated.json'):
                    input_file = os.path.join(lib_path, file_name)
                    self.process_file(input_file, lib_dir)
        
        # Print final statistics
        print("\n" + "=" * 50)
        print("Equivalence evaluation completed statistics:")
        print(f"Total code pairs: {self.total_pairs}")
        print(f"Equivalent code pairs: {self.equivalent_pairs}")
        print(f"Non-equivalent/Error code pairs: {self.error_pairs}")
        print(f"Retention rate: {self.equivalent_pairs/self.total_pairs*100:.2f}%" if self.total_pairs > 0 else "Retention rate: 0%")

def main():
    parser = argparse.ArgumentParser(description='Evaluate equivalence of mutated code pairs')
    parser.add_argument('--input_dir', type=str, default='../codepair_gen/mutated_code_pairs', 
                        help='Directory containing mutated code pairs')
    parser.add_argument('--output_dir', type=str, default='equivalent_code_pairs', 
                        help='Directory to save equivalent code pairs')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Euclidean distance threshold for equivalence determination')
    
    args = parser.parse_args()
    
    # Create and run the evaluator
    evaluator = EquivalenceEvaluator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
    
    evaluator.process_all_files()
    print("\nEquivalence evaluation process completed successfully!")

if __name__ == "__main__":
    main() 