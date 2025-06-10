#!/usr/bin/env python3
"""
Convenient script to extract bug-prone values from bug reports
"""

import os
import sys
import argparse
from pathlib import Path
from bug_report_analyzer import BugReportAnalyzer

def main():
    print("=" * 60)
    print("🐛 Bug-Prone Values Extractor - FUTURE Feedback Component")
    print("=" * 60)
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Extract bug-prone values from bug reports using fine-tuned model')
    parser.add_argument('--model_path', type=str, 
                        default='../2_finetune/fine-tuning/combined_model',
                        help='Path to fine-tuned model (default: ../2_finetune/fine-tuning/combined_model)')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input file containing bug reports (txt or json)')
    parser.add_argument('--input_dir', type=str, default='bug_report',
                        help='Directory containing bug report files (default: bug_report)')
    parser.add_argument('--output_dir', type=str, default='bug_prone_values',
                        help='Output directory for results (default: bug_prone_values)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run model on (default: auto)')
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        print("Please make sure you have fine-tuned a model first.")
        print("You can use the default path or specify --model_path")
        return 1
    
    # Create input directory if it doesn't exist and no input file specified
    if not args.input_file:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"📁 Creating input directory: {input_dir}")
            input_dir.mkdir(parents=True, exist_ok=True)
            print(f"Please place your bug report txt files in: {input_dir}")
            print("Or use --input_file to specify a single file")
            return 1
        
        # Check if directory has supported files
        txt_files = list(input_dir.glob("*.txt"))
        json_files = list(input_dir.glob("*.json"))
        supported_files = txt_files + json_files
        
        if not supported_files:
            print(f"❌ No supported files found in: {input_dir}")
            print("Please add bug report files (.txt or .json)")
            return 1
    
    print(f"🤖 Model path: {args.model_path}")
    if args.input_file:
        print(f"📄 Input file: {args.input_file}")
    else:
        print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"💻 Device: {args.device}")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        print("🔄 Initializing bug report analyzer...")
        analyzer = BugReportAnalyzer(
            model_path=args.model_path,
            device=args.device
        )
        
        # Run analysis
        if args.input_file:
            print(f"🔍 Extracting bug-prone values from: {args.input_file}")
            output_file = os.path.join(args.output_dir, "bug_prone_values.json")
            os.makedirs(args.output_dir, exist_ok=True)
            
            bug_prone_values = analyzer.process_bug_reports_file(
                args.input_file,
                output_file
            )
            
            print("✅ Extraction completed!")
            print(f"🐛 Bug-prone values extracted: {len(bug_prone_values)}")
            print(f"📄 Values saved to: {output_file}")
            
            if bug_prone_values:
                print(f"🔍 Extracted values: {bug_prone_values}")
            else:
                print("⚠️  No bug-prone values found")
        
        else:
            print(f"🔍 Processing directory: {args.input_dir}")
            results = analyzer.process_directory(
                args.input_dir,
                args.output_dir
            )
            
            print("✅ Extraction completed!")
            print(f"📁 Files processed: {len(results)}")
            
            # Calculate totals
            total_values = []
            for file_values in results.values():
                total_values.extend(file_values)
            
            print(f"🐛 Total bug-prone values extracted: {len(total_values)}")
            print(f"📄 Results saved to: {args.output_dir}")
            
            if total_values:
                print(f"🔍 All extracted values: {total_values}")
            else:
                print("⚠️  No bug-prone values found")
        
    except KeyboardInterrupt:
        print("\n❌ Extraction interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return 1
    
    print("\n🎉 Bug-prone value extraction completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())