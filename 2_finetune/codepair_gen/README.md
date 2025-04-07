# Code Pair Generation

This directory contains two main scripts for generating and enhancing code pairs for deep learning APIs:

1. `codepair_gen.py`: Uses a LLM to generate PyTorch and TensorFlow implementations corresponding to source APIs

2. `mutate.py`: Mutates the generated code pairs to increase data diversity

## Model Download

This project uses the CodeLlama-13B model to generate code pairs. You need to download the model:

```bash
# Install huggingface_hub if you haven't already
pip install huggingface_hub

# Download the model
python -c "from huggingface_hub import snapshot_download; snapshot_download('codellama/CodeLlama-13b-Python-hf', local_dir='CodeLlama-13b-Python-hf')"
```

Alternatively, you can download it directly from the Hugging Face website: [CodeLlama-13b-Python-hf](https://huggingface.co/codellama/CodeLlama-13b-Python-hf)


## Usage

### 1. Generate Code Pairs

Run the following command to generate code pairs:

```bash
cd 2_finetune/codepair_gen
python codepair_gen.py
```

Default parameters:
- Input directory: `../api_info_extraction/api_info`
- API list directory: `../api_info_extraction/api_list`
- Output directory: `code_pairs`
- Model path: `../CodeLlama-13b-Python-hf`

To modify these parameters, edit the `main` function in the `codepair_gen.py` file:

### 2. Mutate Code Pairs

Run the following command to mutate the generated code pairs to increase dataset diversity:

```bash
cd 2_finetune/codepair_gen
python mutate.py --mutation_count 20
```

Parameter description:
- `--input_dir`: Directory containing original code pairs (default: `code_pairs`)
- `--output_dir`: Directory to save mutated code pairs (default: `mutated_code_pairs`)
- `--mutation_count`: Number of mutations to create for each original code pair (default: 10)

## Output Format

### Code Pairs (codepair_gen.py)

Generated code pairs will be saved in the `code_pairs/` directory, organized by library:

```
code_pairs/
├── mindspore/
│   ├── mindspore_ops_abs_pairs.json
│   └── ...
├── oneflow/
│   ├── oneflow_zeros_pairs.json
│   └── ...
└── mlx/
    ├── mlx_core_abs_pairs.json
    └── ...
```

Each JSON file contains 5 code examples, with each example containing three implementations: original library, PyTorch, and TensorFlow.

### Mutated Code Pairs (mutate.py)

Mutated code will be saved in the `mutated_code_pairs/` directory, organized by library:

```
mutated_code_pairs/
├── mindspore/
│   ├── mindspore_ops_abs_mutated.json
│   └── ...
├── oneflow/
│   ├── oneflow_zeros_mutated.json
│   └── ...
└── mlx/
    ├── mlx_core_abs_mutated.json
    └── ...
```

Each JSON file contains the original code pairs and multiple mutated versions.
