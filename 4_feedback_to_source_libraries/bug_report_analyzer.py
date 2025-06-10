import os
import json
import torch
import argparse
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import re
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BugReportAnalyzer:
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the Bug Report Analyzer with fine-tuned model
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to load model on
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Load model and tokenizer
        self._load_model()
        
        # Generation configuration
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None
        )
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            logger.info(f"Loading model from {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_prompt(self, bug_reports: List[str]) -> str:
        """
        Create the analysis prompt from bug reports
        
        Args:
            bug_reports: List of bug report strings
            
        Returns:
            Formatted prompt string
        """
        bug_content = ""
        for i, report in enumerate(bug_reports, 1):
            bug_content += f"{{bug report {i}}}\n{report}\n\n"
        
        prompt = f"""**Task**: Analyze the target library bug reports below and summarize a bug-prone API input meeting these criteria:
1. Must be DIRECTLY associated with the API call causing the bug
2. Only summarize the result as an ONE-DIMENSIONAL arrays/tensors

**Bug Report Content**:
{bug_content.strip()}

**Output Format**: Only return the values in an one-dimensional array,
[bug-prone values]"""
        
        return prompt
    
    def analyze_bug_reports(self, bug_reports: List[str]) -> str:
        """
        Analyze bug reports and extract bug-prone API inputs
        
        Args:
            bug_reports: List of bug report strings
            
        Returns:
            Generated analysis result
        """
        try:
            # Create prompt
            prompt = self.create_prompt(bug_reports)
            logger.info("Created analysis prompt")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            if torch.cuda.is_available() and self.device == "auto":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            logger.info("Generating analysis...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    use_cache=True
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            ).strip()
            
            logger.info("Analysis completed")
            return response
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise
    
    def extract_bug_prone_values(self, analysis_result: str) -> List[Any]:
        """
        Extract bug-prone values from analysis result
        
        Args:
            analysis_result: Generated analysis text
            
        Returns:
            List of extracted bug-prone values
        """
        try:
            # Look for array patterns in the result
            patterns = [
                r'\[(.*?)\]',  # Standard array format [1, 2, 3]
                r'bug-prone values:\s*\[(.*?)\]',  # Specific format
                r'values:\s*\[(.*?)\]',  # General values format
            ]
            
            for pattern in patterns:
                matches = re.search(pattern, analysis_result, re.IGNORECASE | re.DOTALL)
                if matches:
                    values_str = matches.group(1).strip()
                    if values_str:
                        # Parse the values
                        try:
                            # Try to evaluate as Python literal
                            values = eval(f"[{values_str}]")
                            return values
                        except:
                            # If evaluation fails, split by comma
                            values = [v.strip().strip("'\"") for v in values_str.split(',')]
                            return [v for v in values if v]  # Remove empty values
            
            # If no pattern matches, return empty list
            logger.warning("Could not extract bug-prone values from analysis result")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting bug-prone values: {e}")
            return []
    
    def load_bug_reports_from_file(self, input_file: str) -> List[str]:
        """
        Load bug reports from a file (supports both txt and json formats)
        
        Args:
            input_file: Path to input file containing bug reports
            
        Returns:
            List of bug report strings
        """
        try:
            file_path = Path(input_file)
            
            if file_path.suffix.lower() == '.txt':
                # Read txt file
                with open(input_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # If file contains multiple reports separated by newlines or specific markers
                if '\n---\n' in content:
                    # Split by separator
                    bug_reports = [report.strip() for report in content.split('\n---\n') if report.strip()]
                elif content.count('\n\n') > 0:
                    # Split by double newlines
                    bug_reports = [report.strip() for report in content.split('\n\n') if report.strip()]
                else:
                    # Treat entire file as single bug report
                    bug_reports = [content]
                    
            elif file_path.suffix.lower() == '.json':
                # Read json file (legacy support)
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    bug_reports = [str(report) for report in data]
                elif isinstance(data, dict):
                    if 'bug_reports' in data:
                        bug_reports = [str(report) for report in data['bug_reports']]
                    elif 'reports' in data:
                        bug_reports = [str(report) for report in data['reports']]
                    else:
                        bug_reports = [str(v) for v in data.values() if v]
                else:
                    bug_reports = [str(data)]
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            return bug_reports
            
        except Exception as e:
            logger.error(f"Error loading bug reports from {input_file}: {e}")
            raise
    
    def process_bug_reports_file(self, input_file: str, output_file: str = None) -> List[Any]:
        """
        Process bug reports from a file
        
        Args:
            input_file: Path to input file containing bug reports
            output_file: Path to output JSON file (optional)
            
        Returns:
            List of extracted bug-prone values
        """
        try:
            # Load bug reports
            logger.info(f"Loading bug reports from {input_file}")
            bug_reports = self.load_bug_reports_from_file(input_file)
            
            if not bug_reports:
                logger.warning("No bug reports found in input file")
                return []
            
            logger.info(f"Found {len(bug_reports)} bug reports")
            
            # Analyze bug reports
            analysis_result = self.analyze_bug_reports(bug_reports)
            
            # Extract bug-prone values
            bug_prone_values = self.extract_bug_prone_values(analysis_result)
            
            # Save results if output file specified
            if output_file:
                logger.info(f"Saving bug-prone values to {output_file}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(bug_prone_values, f, indent=2, ensure_ascii=False)
            
            return bug_prone_values
            
        except Exception as e:
            logger.error(f"Error processing bug reports file: {e}")
            raise
    
    def process_directory(self, input_dir: str, output_dir: str = None) -> Dict[str, List[Any]]:
        """
        Process all bug report files in a directory
        
        Args:
            input_dir: Directory containing bug report files
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary mapping file names to their extracted bug-prone values
        """
        try:
            input_path = Path(input_dir)
            if not input_path.exists():
                raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
            
            # Create output directory if specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            
            all_results = {}
            
            # Process all txt and json files in directory
            supported_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.json"))
            logger.info(f"Found {len(supported_files)} files to process")
            
            for input_file in supported_files:
                logger.info(f"Processing {input_file.name}")
                
                # Determine output file path
                if output_dir:
                    output_file = output_path / f"{input_file.stem}_values.json"
                else:
                    output_file = None
                
                # Process the file
                bug_prone_values = self.process_bug_reports_file(str(input_file), str(output_file) if output_file else None)
                all_results[input_file.name] = bug_prone_values
            
            # Save combined results
            if output_dir:
                combined_file = output_path / "all_bug_prone_values.json"
                with open(combined_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Combined results saved to {combined_file}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error processing directory: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Extract bug-prone values from bug reports using fine-tuned model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input file containing bug reports (txt or json)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing bug report files')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSON file for bug-prone values')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run model on')
    
    args = parser.parse_args()
    
    if not args.input_file and not args.input_dir:
        parser.error("Either --input_file or --input_dir must be specified")
    
    # Initialize analyzer
    analyzer = BugReportAnalyzer(
        model_path=args.model_path,
        device=args.device
    )
    
    try:
        if args.input_file:
            # Process single file
            bug_prone_values = analyzer.process_bug_reports_file(
                args.input_file,
                args.output_file
            )
            print("Bug-prone values extracted:")
            print(bug_prone_values)
            
        elif args.input_dir:
            # Process directory
            results = analyzer.process_directory(
                args.input_dir,
                args.output_dir
            )
            
            # Print summary
            total_values = []
            for file_results in results.values():
                total_values.extend(file_results)
            
            print(f"Processed {len(results)} files")
            print(f"Total bug-prone values extracted: {len(total_values)}")
            print("All bug-prone values:", total_values)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    logger.info("Bug-prone value extraction completed successfully")
    return 0

if __name__ == "__main__":
    exit(main()) 