import os
import json
import glob
import argparse
from typing import Dict, List, Any, Tuple

class DatasetConstructor:
    def __init__(
        self, 
        input_dir: str, 
        output_dir: str,
        framework_mapping: Dict[str, str] = None
    ):
        """
        Initialize dataset constructor
        
        Args:
            input_dir: Directory containing mutated code pairs
            output_dir: Directory to save constructed datasets
            framework_mapping: Mapping from library names to framework names
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Default framework mappings if not provided
        if framework_mapping is None:
            self.framework_mapping = {
                "mindspore": "MindSpore",
                "oneflow": "OneFlow",
                "mlx": "MLX"
            }
        else:
            self.framework_mapping = framework_mapping
        
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
    
    def process_files(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Process all mutated code pair files
        
        Returns:
            Tuple containing conversion dataset and generation dataset
        """
        conversion_dataset = []
        generation_dataset = []
        
        # Process each library
        for lib_dir in os.listdir(self.input_dir):
            lib_path = os.path.join(self.input_dir, lib_dir)
            
            if not os.path.isdir(lib_path):
                continue
            
            print(f"Processing {lib_dir} library...")
            
            # Get framework name for this library
            framework_name = self.framework_mapping.get(lib_dir, lib_dir.capitalize())
            
            # Process each JSON file in the library directory
            for file_path in glob.glob(os.path.join(lib_path, "*_mutated.json")):
                # Extract API name from file name
                file_name = os.path.basename(file_path)
                api_name = file_name.replace("_mutated.json", "").replace("_", ".")
                
                # Process file
                file_conversion, file_generation = self._process_file(
                    file_path, api_name, framework_name
                )
                
                conversion_dataset.extend(file_conversion)
                generation_dataset.extend(file_generation)
                
                print(f"  Processed {file_name}: {len(file_conversion)} conversion examples, {len(file_generation)} generation examples")
        
        return conversion_dataset, generation_dataset
    
    def _process_file(
        self, 
        file_path: str, 
        api_name: str, 
        framework_name: str
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Process a single mutated code pair file
        
        Args:
            file_path: Path to the mutated code pair file
            api_name: Name of the API
            framework_name: Name of the framework
            
        Returns:
            Tuple containing conversion examples and generation examples
        """
        conversion_examples = []
        generation_examples = []
        
        # Load code pairs
        with open(file_path, 'r') as f:
            code_pairs = json.load(f)
        
        # Process each code pair
        for code_pair in code_pairs:
            if not isinstance(code_pair, dict) or not all(k in code_pair for k in ["original", "pytorch", "tensorflow"]):
                # Skip invalid code pairs
                continue
            
            # Create conversion examples (PyTorch → Original, TensorFlow → Original)
            if "pytorch" in code_pair and code_pair["pytorch"].strip() and "original" in code_pair and code_pair["original"].strip():
                conversion_examples.append({
                    "seed": code_pair["pytorch"],
                    "problem": f"Convert this code to code that uses the {framework_name} framework",
                    "solution": code_pair["original"]
                })
            
            if "tensorflow" in code_pair and code_pair["tensorflow"].strip() and "original" in code_pair and code_pair["original"].strip():
                conversion_examples.append({
                    "seed": code_pair["tensorflow"],
                    "problem": f"Convert this code to code that uses the {framework_name} framework",
                    "solution": code_pair["original"]
                })
            
            # Create generation example
            if "original" in code_pair and code_pair["original"].strip():
                generation_examples.append({
                    "INSTRUCTION": f"Generate code that calls the '{api_name}' API",
                    "RESPONSE": code_pair["original"]
                })
        
        return conversion_examples, generation_examples
    
    def save_datasets(self, conversion_dataset: List[Dict[str, str]], generation_dataset: List[Dict[str, str]]):
        """
        Save datasets to files
        
        Args:
            conversion_dataset: List of conversion examples
            generation_dataset: List of generation examples
        """
        # Save conversion dataset
        conversion_path = os.path.join(self.output_dir, "conversion_dataset.json")
        with open(conversion_path, 'w') as f:
            json.dump(conversion_dataset, f, indent=2)
        
        # Save generation dataset
        generation_path = os.path.join(self.output_dir, "generation_dataset.json")
        with open(generation_path, 'w') as f:
            json.dump(generation_dataset, f, indent=2)
        
        # Save combined dataset (simply concatenate without format conversion)
        combined_dataset = conversion_dataset + generation_dataset
        
        combined_path = os.path.join(self.output_dir, "combined_dataset.json")
        with open(combined_path, 'w') as f:
            json.dump(combined_dataset, f, indent=2)
        
        print(f"\nDatasets saved to:")
        print(f"  Conversion dataset: {conversion_path} ({len(conversion_dataset)} examples)")
        print(f"  Generation dataset: {generation_path} ({len(generation_dataset)} examples)")
        print(f"  Combined dataset: {combined_path} ({len(combined_dataset)} examples)")

def main():
    parser = argparse.ArgumentParser(description='Construct fine-tuning datasets from code pairs')
    parser.add_argument('--input_dir', type=str, default='../codepair_gen/mutated_code_pairs',
                        help='Directory containing mutated code pairs')
    parser.add_argument('--output_dir', type=str, default='datasets',
                        help='Directory to save constructed datasets')
    
    args = parser.parse_args()
    
    # Create and run dataset constructor
    constructor = DatasetConstructor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    conversion_dataset, generation_dataset = constructor.process_files()
    constructor.save_datasets(conversion_dataset, generation_dataset)
    
    print("\nDataset construction completed successfully!")

if __name__ == "__main__":
    main()
