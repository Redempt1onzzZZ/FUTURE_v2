import os
import re
import json
import random
import numpy as np
from typing import List, Dict, Union, Any
import argparse

class CodePairMutator:
    def __init__(self, input_dir: str, output_dir: str, mutation_count: int = 10):
        """
        Initialize the code pair mutator
        
        Args:
            input_dir: Directory containing the generated code pairs
            output_dir: Directory to save mutated code pairs
            mutation_count: Number of mutations to create for each code pair
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mutation_count = mutation_count
        
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
    
    def format_number(self, num: Union[int, float, str]) -> str:
        """Format a number for code insertion"""
        if isinstance(num, str):
            return num
        if np.isnan(num): 
            return "float('nan')"
        elif num == float('inf'):
            return "float('inf')"
        elif num == float('-inf'):
            return "float('-inf')"
        elif isinstance(num, int) or num.is_integer():
            return str(int(num))
        else:
            return str(num)
    
    def parse_arrays(self, code: str) -> List[Dict[str, Any]]:
        """
        Find all arrays in the code and parse their values
        
        Args:
            code: Python code as string
            
        Returns:
            List of dictionaries containing array information
        """
        # Find all arrays in the code
        arrays = []
        # Look for array patterns like [1, 2, 3] or np.array([1, 2, 3])
        array_patterns = [
            r'\[([-\w.\'(), ]+)\]',  # Regular arrays [1, 2, 3]
            r'np\.array\(\[([-\w.\'(), ]+)\]\)',  # NumPy arrays np.array([1, 2, 3])
            r'torch\.tensor\(\[([-\w.\'(), ]+)\]\)',  # PyTorch tensors torch.tensor([1, 2, 3])
            r'tf\.constant\(\[([-\w.\'(), ]+)\]\)',  # TensorFlow constants tf.constant([1, 2, 3])
            r'mx\.array\(\[([-\w.\'(), ]+)\]\)'  # MLX arrays mx.array([1, 2, 3])
        ]
        
        for pattern in array_patterns:
            for match in re.finditer(pattern, code):
                full_match = match.group(0)
                array_content = match.group(1)
                
                # Parse array values
                values = []
                for item in array_content.split(','):
                    item = item.strip()
                    if not item:
                        continue
                    if item == "float('nan')":
                        values.append(float('nan'))
                    elif item == "float('inf')":
                        values.append(float('inf'))
                    elif item == "float('-inf')":
                        values.append(float('-inf'))
                    else:
                        try:
                            # Try to convert to number
                            values.append(float(item))
                        except ValueError:
                            # Keep as string if not a number
                            values.append(item)
                
                if values:  # Only add non-empty arrays
                    arrays.append({
                        'full_match': full_match,
                        'content': array_content,
                        'values': values,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return arrays
    
    def mutate_array(self, array_info: Dict[str, Any], mutation_index: int) -> str:
        """
        Create a mutated version of an array
        
        Args:
            array_info: Dictionary containing array information
            mutation_index: Index for this mutation (used to create deterministic variations)
            
        Returns:
            String with the mutated array
        """
        values = array_info['values']
        if not values:
            return array_info['full_match']
        
        # Create a mutated copy of the values
        new_values = values.copy()
        
        # Mutation strategies
        mutation_type = mutation_index % 5
        
        if mutation_type == 0:
            # Strategy 1: Change one value based on position
            if len(new_values) > 0:
                pos = mutation_index % len(new_values)
                if isinstance(new_values[pos], (int, float)) and not np.isnan(new_values[pos]) and new_values[pos] != float('inf') and new_values[pos] != float('-inf'):
                    new_values[pos] = new_values[pos] + mutation_index + 1
        
        elif mutation_type == 1:
            # Strategy 2: Scale all numeric values
            scale_factor = (mutation_index % 5) + 2  # Scale by 2, 3, 4, 5, or 6
            for i, val in enumerate(new_values):
                if isinstance(val, (int, float)) and not np.isnan(val) and val != float('inf') and val != float('-inf'):
                    new_values[i] = val * scale_factor
        
        elif mutation_type == 2:
            # Strategy 3: Add a constant to all numeric values
            add_value = mutation_index + 1
            for i, val in enumerate(new_values):
                if isinstance(val, (int, float)) and not np.isnan(val) and val != float('inf') and val != float('-inf'):
                    new_values[i] = val + add_value
        
        elif mutation_type == 3:
            # Strategy 4: Replace with special values (NaN, Inf)
            if len(new_values) > 0:
                special_values = [float('nan'), float('inf'), float('-inf')]
                pos = mutation_index % len(new_values)
                if isinstance(new_values[pos], (int, float)):
                    new_values[pos] = special_values[(mutation_index // len(new_values)) % len(special_values)]
        
        else:
            # Strategy 5: Change array shape (add elements)
            for _ in range(min(3, mutation_index % 5 + 1)):
                if all(isinstance(v, (int, float)) for v in new_values):
                    # If all elements are numeric, add a numeric value
                    new_values.append(mutation_index + 10)
                else:
                    # Otherwise add a string
                    new_values.append(f"'item{mutation_index}'")
        
        # Format the array back to a string
        array_str = '[' + ', '.join(self.format_number(val) for val in new_values) + ']'
        
        # Reconstruct the full pattern (np.array, torch.tensor, etc.)
        if array_info['full_match'].startswith('np.array'):
            return f"np.array({array_str})"
        elif array_info['full_match'].startswith('torch.tensor'):
            return f"torch.tensor({array_str})"
        elif array_info['full_match'].startswith('tf.constant'):
            return f"tf.constant({array_str})"
        elif array_info['full_match'].startswith('mx.array'):
            return f"mx.array({array_str})"
        else:
            return array_str
    
    def mutate_code(self, code: str, mutation_index: int) -> str:
        """
        Create a mutated version of the code
        
        Args:
            code: Original code as string
            mutation_index: Index for this mutation
            
        Returns:
            Mutated code as string
        """
        # Parse arrays in the code
        arrays = self.parse_arrays(code)
        
        if not arrays:
            # If no arrays found, make minor changes to numeric literals
            # Find numeric literals in the code
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', code)
            if numbers:
                for num in numbers:
                    try:
                        value = float(num)
                        # Skip if special value
                        if np.isnan(value) or value == float('inf') or value == float('-inf'):
                            continue
                        # Create new value
                        new_value = value + mutation_index + 1
                        # Replace in code (be careful to replace only exact matches)
                        code = re.sub(r'\b' + num + r'\b', str(new_value), code)
                        # Only mutate one number to keep changes minimal
                        break
                    except ValueError:
                        continue
            return code
        
        # Select an array to mutate based on mutation_index
        array_to_mutate = arrays[mutation_index % len(arrays)]
        
        # Mutate the array
        mutated_array = self.mutate_array(array_to_mutate, mutation_index)
        
        # Replace the original array with the mutated one
        parts = [
            code[:array_to_mutate['start']],
            mutated_array,
            code[array_to_mutate['end']:]
        ]
        return ''.join(parts)
    
    def mutate_code_pair(self, code_pair: Dict[str, str], mutation_index: int) -> Dict[str, str]:
        """
        Create a mutated version of a code pair
        
        Args:
            code_pair: Dictionary with original, pytorch, and tensorflow implementations
            mutation_index: Index for this mutation
            
        Returns:
            Dictionary with mutated implementations
        """
        mutated_pair = {}
        
        # Mutate each implementation consistently
        for impl_type in ['original', 'pytorch', 'tensorflow']:
            if impl_type in code_pair:
                mutated_pair[impl_type] = self.mutate_code(code_pair[impl_type], mutation_index)
        
        return mutated_pair
    
    def process_file(self, input_file: str, lib_name: str):
        """
        Process a single code pair file
        
        Args:
            input_file: Path to the input JSON file
            lib_name: Library name (for output directory structure)
        """
        # Create output directory
        lib_output_dir = os.path.join(self.output_dir, lib_name)
        os.makedirs(lib_output_dir, exist_ok=True)
        
        # Load code pairs
        with open(input_file, 'r') as f:
            code_pairs = json.load(f)
        
        # Skip if no code pairs
        if not code_pairs:
            print(f"No code pairs found in {input_file}, skipping")
            return
        
        # Get API name from file name
        api_name = os.path.basename(input_file).replace('_pairs.json', '')
        
        # Process each original code pair
        all_mutated_pairs = []
        
        # Keep original pairs
        all_mutated_pairs.extend(code_pairs)
        
        # Generate mutations
        for i in range(self.mutation_count):
            for j, code_pair in enumerate(code_pairs):
                mutated_pair = self.mutate_code_pair(code_pair, i * len(code_pairs) + j)
                all_mutated_pairs.append(mutated_pair)
        
        # Save mutated code pairs
        output_file = os.path.join(lib_output_dir, f"{api_name}_mutated.json")
        with open(output_file, 'w') as f:
            json.dump(all_mutated_pairs, f, indent=2)
        
        print(f"Created {len(all_mutated_pairs)} code pairs for {api_name} (Original: {len(code_pairs)}, Mutated: {len(all_mutated_pairs) - len(code_pairs)})")
    
    def process_all_files(self):
        """Process all code pair files in the input directory"""
        for lib_dir in os.listdir(self.input_dir):
            lib_path = os.path.join(self.input_dir, lib_dir)
            
            if not os.path.isdir(lib_path):
                continue
            
            print(f"\nProcessing {lib_dir} library...")
            
            # Process each JSON file in the library directory
            for file_name in os.listdir(lib_path):
                if file_name.endswith('_pairs.json'):
                    input_file = os.path.join(lib_path, file_name)
                    self.process_file(input_file, lib_dir)

def main():
    parser = argparse.ArgumentParser(description='Mutate code pairs to increase dataset size')
    parser.add_argument('--input_dir', type=str, default='code_pairs', 
                        help='Directory containing generated code pairs')
    parser.add_argument('--output_dir', type=str, default='mutated_code_pairs', 
                        help='Directory to save mutated code pairs')
    parser.add_argument('--mutation_count', type=int, default=100,
                        help='Number of mutations to create for each code pair')
    
    args = parser.parse_args()
    
    # Create and run the mutator
    mutator = CodePairMutator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mutation_count=args.mutation_count
    )
    
    mutator.process_all_files()
    print("\nMutation process completed successfully!")

if __name__ == "__main__":
    main()
