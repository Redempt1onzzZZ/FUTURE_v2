#!/usr/bin/env python3
"""
Script to run equivalence evaluation
"""

import os
import sys
from equivalence_evaluation import EquivalenceEvaluator

def main():
    # Set paths - currently in equivalence_evaluation folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input directory: point to mutated code pairs folder (relative path)
    input_dir = os.path.join(current_dir, '../codepair_gen/mutated_code_pairs')
    
    # Output directory: save equivalent code pairs in current folder
    output_dir = os.path.join(current_dir, 'equivalent_code_pairs')
    
    # Distance threshold
    threshold = 0.01
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        print("Please run mutate.py first to generate mutated code pairs")
        return
    
    print("=" * 60)
    print("🔍 Equivalence Evaluation Component")
    print("=" * 60)
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📏 Distance threshold: {threshold}")
    print("=" * 60)
    
    # Create evaluator
    evaluator = EquivalenceEvaluator(
        input_dir=input_dir,
        output_dir=output_dir,
        threshold=threshold
    )
    
    # Run evaluation
    try:
        evaluator.process_all_files()
        
        print("\n✅ Equivalence evaluation completed!")
        print(f"📂 Results saved to: {output_dir}")
        
        # Display statistics
        if evaluator.total_pairs > 0:
            print(f"\n📊 Statistics:")
            print(f"   📝 Total code pairs: {evaluator.total_pairs}")
            print(f"   ✅ Equivalent code pairs: {evaluator.equivalent_pairs}")
            print(f"   ❌ Non-equivalent code pairs: {evaluator.error_pairs}")
            print(f"   📈 Retention rate: {evaluator.equivalent_pairs/evaluator.total_pairs*100:.2f}%")
        
    except KeyboardInterrupt:
        print("\n❌ User interrupted the evaluation process")
    except Exception as e:
        print(f"\n❌ Error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 