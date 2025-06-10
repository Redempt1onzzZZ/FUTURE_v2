# Equivalence Evaluator

This component evaluates the equivalence of mutated code pairs by running the code and calculating the Euclidean distance between results to determine whether to retain code pairs.

## File Structure

```
equivalence_evaluation/
├── equivalence_evaluation.py  # Main equivalence evaluation class
├── run_evaluation.py         # Execution script
└── README.md                 # Documentation
```

## Usage

### 1. Direct execution
```bash
cd 2_finetune/equivalence_evaluation
python run_evaluation.py
```

### 2. Command line arguments
```bash
python equivalence_evaluation.py --input_dir ../codepair_gen/mutated_code_pairs --output_dir equivalent_code_pairs --threshold 0.01
```

## Parameters

- `--input_dir`: Input directory containing mutated code pairs
- `--output_dir`: Output directory to save equivalent code pairs
- `--threshold`: Euclidean distance threshold for equivalence determination (default: 0.01)