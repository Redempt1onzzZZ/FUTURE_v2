# Historical Bug Collection

This code implements the first part of FUTURE, which focuses on collecting historical bug codes from existing DL libraries.

## Description

This spider performs the following functions:
- Automatically crawl issues (label specific to bugs) from TensorFlow and PyTorch repositories
- Extracts reproduction code snippets from these issues
- Saves the code in a standardized format to the local file system

## Output

The scraped code snippets are organized as follows:
```
Github_Issues_code/
├── issue_id_title1/
│   └── code_1.py
├── issue_id_title2/
│   └── code_1.py
└── ...
```

## Note
Remember to replace the path in the code with your own local path and adjust the url in the code to crawl label-specific issues from existing DL libraries.