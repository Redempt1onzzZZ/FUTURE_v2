# Dataset Construction for Fine-tuning

This tool constructs fine-tuning datasets from code pairs generated in the previous steps. It creates three datasets:

1. **Conversion Dataset**: For translating PyTorch/TensorFlow code to target frameworks
2. **Generation Dataset**: For generating code using specific APIs
3. **Combined Dataset**: Merging both datasets

## Usage

Run the script with:

```bash
python dataset.py [--input_dir PATH] [--output_dir PATH]
```

### Arguments

- `--input_dir`: Directory containing mutated code pairs (default: `../codepair_gen/mutated_code_pairs`)
- `--output_dir`: Directory to save constructed datasets (default: `datasets`)

## Output Format

### Conversion Dataset

Saved as `conversion_dataset.json`:

```json
[
  {
    "seed": "<pytorch or tensorflow code>",
    "problem": "Convert this code to code that uses the <Framework> framework",
    "solution": "<original code>"
  },
  ...
]
```

### Generation Dataset

Saved as `generation_dataset.json`:

```json
[
  {
    "INSTRUCTION": "Generate code that calls the '<api_name>' API",
    "RESPONSE": "<original code>"
  },
  ...
]
```

### Combined Dataset

Saved as `combined_dataset.json`:

```json
[
  {
    "INSTRUCTION": "<instruction from conversion or generation dataset>",
    "RESPONSE": "<code solution>"
  },
  ...
]
```
