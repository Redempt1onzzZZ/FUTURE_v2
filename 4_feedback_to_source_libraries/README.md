# Feedback to Source Libraries

This component extracts bug-prone API input values from bug reports discovered by FUTURE using fine-tuned models. The extracted values are formatted as one-dimensional arrays that can be fed back to source libraries for testing and validation.

## File Structure

```
4_feedback_to_source_libraries/
├── bug_report_analyzer.py  # Main extractor class
├── run_analysis.py         # Convenient execution script
├── README.md              # Documentation
├── bug_report/            # Input directory for bug reports (txt files)
└── bug_prone_values/      # Output directory for results (created automatically)
```

## Usage

### 1. Quick Start

Place your bug report txt files in the `bug_report` directory and run:

```bash
cd 4_feedback_to_source_libraries

# Extract values from all txt files in bug_report directory
python run_analysis.py
```

## Input Format

Bug reports should be provided as txt files. The extractor supports multiple formats:

### Format 1: Single bug report per file
```
NumPy array operation failed with shape mismatch error when using np.dot() with arrays of shape (3, 4) and (5, 3). The function expected compatible dimensions but received incompatible shapes causing ValueError.
```

### Format 2: Multiple reports separated by double newlines
```
Bug report 1 text here...

Bug report 2 text here...

Bug report 3 text here...
```

### Format 3: Multiple reports separated by `---`
```
Bug report 1 text here...
---
Bug report 2 text here...
---
Bug report 3 text here...
```

### Format 4: JSON format (legacy support)
```json
{
  "bug_reports": [
    "Bug report 1 text...",
    "Bug report 2 text..."
  ]
}
```

## Prompt Template

The extractor uses this prompt template to query the fine-tuned model:

```
**Task**: Analyze the target library bug reports below and summarize a bug-prone API input meeting these criteria:
1. Must be DIRECTLY associated with the API call causing the bug
2. Only summarize the result as an ONE-DIMENSIONAL arrays/tensors

**Bug Report Content**:
{bug report 1}
{bug report 2}
...
{bug report n}

**Output Format**: Only return the values in an one-dimensional array,
[bug-prone values]
```
