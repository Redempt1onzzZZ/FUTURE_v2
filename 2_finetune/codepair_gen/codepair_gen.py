import json
import os
import time
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CodePairGenerator:
    def __init__(self, api_info_dir: str, api_list_dir: str, output_dir: str, model_path: str):
        """
        Initialize the code pair generator
        
        Args:
            api_info_dir: Directory containing the API info JSON files
            api_list_dir: Directory containing the API list files
            output_dir: Directory to save generated code pairs
            model_path: Path to the CodeLlama model
        """
        self.api_info_dir = api_info_dir
        self.api_list_dir = api_list_dir
        self.output_dir = output_dir
        self.api_info: Dict[str, List[Dict]] = {}
        self.api_lists: Dict[str, List[str]] = {}
        
        # Load model and tokenizer
        print("Loading CodeLlama model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully")
        
        self._load_api_info()
        self._load_api_lists()
        
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
    def _load_api_info(self):
        """Load API information from JSON files"""
        for filename in os.listdir(self.api_info_dir):
            if filename.endswith('_api_info.json'):
                lib_name = filename.replace('_api_info.json', '')
                with open(os.path.join(self.api_info_dir, filename), 'r') as f:
                    self.api_info[lib_name] = json.load(f)
                    
    def _load_api_lists(self):
        """Load API lists from txt files"""
        for filename in os.listdir(self.api_list_dir):
            if filename.endswith('_api.txt'):
                lib_name = filename.replace('_api.txt', '')
                with open(os.path.join(self.api_list_dir, filename), 'r') as f:
                    self.api_lists[lib_name] = [line.strip() for line in f if line.strip()]
    
    def _construct_prompt(self, lib_name: str, api_name: str, api_info: Dict) -> str:
        """
        Construct prompt for CodeLlama
        
        Args:
            lib_name: Library name (mindspore/oneflow/mlx)
            api_name: API name
            api_info: API information dictionary
            
        Returns:
            Constructed prompt string
        """
        prompt = f"# Specify the Task\n"
        prompt += f"You are a code libraries converter and generator. I'll give you the "
        prompt += f"API documentation and code examples of {api_name} from the {lib_name} library. Your "
        prompt += f"task is to provide me 5 different code pairs using this {lib_name} API and its equivalent "
        prompt += f"implementations in PyTorch and TensorFlow. Generate diverse examples covering different use cases "
        prompt += f"and edge cases like Nan, Inf, and various input shapes and data types.\n\n"
        
        prompt += f"# API Documentation\n"
        prompt += f"{api_name}\n"
        
        if 'signature' in api_info:
            prompt += f"{api_info['signature']}\n"
            
        if 'description' in api_info:
            prompt += f"{api_info['description']}\n"
            
        if 'parameters' in api_info:
            prompt += f"\nParameters:\n{api_info['parameters']}\n"
            
        if 'returns' in api_info:
            prompt += f"\nReturns:\n{api_info['returns']}\n"
            
        if 'examples' in api_info and api_info['examples']:
            prompt += f"\n# Code Examples\n{api_info['examples']}\n"
            
        prompt += "\nPlease provide 5 different code examples in the following format:\n"
        prompt += "## Example 1\n"
        prompt += "```python\n# Original Implementation\n<code using original API>\n\n"
        prompt += "# PyTorch Implementation\n<equivalent PyTorch code>\n\n"
        prompt += "# TensorFlow Implementation\n<equivalent TensorFlow code>\n```\n\n"
        prompt += "## Example 2\n... and so on for all 5 examples."
        
        return prompt
        
    def generate_code_pairs(self, lib_name: str, api_name: str) -> Optional[List[Dict[str, str]]]:
        """
        Generate code pairs for a specific API
        
        Args:
            lib_name: Library name (mindspore/oneflow/mlx)
            api_name: API name
            
        Returns:
            List of dictionaries containing original, PyTorch and TensorFlow implementations
        """
        # Find API info
        api_info = None
        for info in self.api_info.get(lib_name, []):
            if info.get('name') == api_name:
                api_info = info
                break
                
        if not api_info:
            print(f"API information not found for {api_name}")
            return None
            
        # Construct prompt
        prompt = self._construct_prompt(lib_name, api_name, api_info)
        
        try:
            # Generate code using CodeLlama
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=2000,  
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            code_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            code_response = code_response[len(prompt):]
            
            # Parse code blocks for multiple examples
            all_code_pairs = []
            current_example = {}
            current_impl = None
            current_code = []
            in_example = False
            
            for line in code_response.split('\n'):
                # Check for new example
                if line.strip().startswith('## Example'):
                    # Save previous example if it exists
                    if current_impl and current_code:
                        current_example[current_impl] = '\n'.join(current_code).strip()
                    
                    if current_example and 'original' in current_example and 'pytorch' in current_example and 'tensorflow' in current_example:
                        all_code_pairs.append(current_example)
                    
                    # Reset for new example
                    current_example = {}
                    current_impl = None
                    current_code = []
                    in_example = True
                    continue
                
                # Process implementations within an example
                if in_example:
                    if '# Original Implementation' in line:
                        if current_impl and current_code:
                            current_example[current_impl] = '\n'.join(current_code).strip()
                        current_impl = 'original'
                        current_code = []
                    elif '# PyTorch Implementation' in line:
                        if current_impl and current_code:
                            current_example[current_impl] = '\n'.join(current_code).strip()
                        current_impl = 'pytorch'
                        current_code = []
                    elif '# TensorFlow Implementation' in line:
                        if current_impl and current_code:
                            current_example[current_impl] = '\n'.join(current_code).strip()
                        current_impl = 'tensorflow'
                        current_code = []
                    elif line.strip() and current_impl:
                        if not (line.startswith('```') or line.endswith('```')):
                            current_code.append(line)
            
            # Save the last implementation and example
            if current_impl and current_code:
                current_example[current_impl] = '\n'.join(current_code).strip()
            
            if current_example and 'original' in current_example and 'pytorch' in current_example and 'tensorflow' in current_example:
                all_code_pairs.append(current_example)
            
            # Ensure we have up to 5 examples
            return all_code_pairs[:5]
            
        except Exception as e:
            print(f"Error generating code pairs for {api_name}: {str(e)}")
            return None
            
    def generate_all_code_pairs(self):
        """Generate code pairs for all APIs in the lists"""
        for lib_name, api_list in self.api_lists.items():
            print(f"\nProcessing {lib_name} APIs...")
            
            # Create library-specific output directory
            lib_output_dir = os.path.join(self.output_dir, lib_name)
            os.makedirs(lib_output_dir, exist_ok=True)
            
            # Process each API
            for i, api_name in enumerate(api_list, 1):
                print(f"[{i}/{len(api_list)}] Generating code pairs for {api_name}")
                
                # Generate code pairs
                all_code_pairs = self.generate_code_pairs(lib_name, api_name)
                
                if all_code_pairs:
                    # Save to file
                    output_file = os.path.join(lib_output_dir, f"{api_name.replace('.', '_')}_pairs.json")
                    with open(output_file, 'w') as f:
                        json.dump(all_code_pairs, f, indent=2)
                    print(f"Saved {len(all_code_pairs)} code pairs to {output_file}")
                else:
                    print(f"Failed to generate code pairs for {api_name}")
                
                # Add delay to avoid resource exhaustion
                time.sleep(1)

def main():
    # Initialize generator
    generator = CodePairGenerator(
        api_info_dir="../api_info_extraction/api_info",
        api_list_dir="../api_info_extraction/api_list",
        output_dir="code_pairs",
        model_path="../2_finetune/CodeLlama-13b-Python-hf"
    )
    
    # Generate code pairs for all APIs
    generator.generate_all_code_pairs()

if __name__ == "__main__":
    main()
