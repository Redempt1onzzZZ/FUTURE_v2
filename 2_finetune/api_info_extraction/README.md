# API Information Extraction for DL Libraries

This directory contains scripts for extracting API information from different DL libraries' documentation. The scripts are designed to scrape detailed information including function signatures, descriptions, parameters, return values, and examples from the official documentation of each library.

In the codes, we use the latest URL of each library's official documentation. If you want to use the older version of the documentation, you can modify the URL in the script.

## Usage

1. Prepare API list files:
   - Create a text file (e.g., `mindspore_api.txt`) containing one API name per line
   - Place the file in the `api_list` directory

2. Run the extraction script:
   ```bash
   python mindspore.py  # For MindSpore
   python oneflow.py    # For OneFlow
   python mlx.py        # For MLX
   ```

3. Output:
   - The extracted information will be saved in JSON format in the `api_info` directory
   - Each library will have its own output file (e.g., `mindspore_api_info.json`)

## Important Notes

1. **Documentation Structure Differences**:
   - Each DL library has its own documentation structure
   - The extraction scripts are specifically designed for their respective libraries
   - When testing with other DL libraries, you need to modify the code to match their documentation structure

2. **Required Modifications for New Libraries**:
   - Update the base URL and URL construction logic
   - Modify HTML parsing rules to match the new documentation structure
   - Adjust the extraction logic for signatures, descriptions, parameters, etc.

3. **Dependencies**:
   - requests
   - beautifulsoup4

## Error Handling

The scripts include error handling for:
- Failed HTTP requests
- Missing documentation pages
- Malformed HTML content
- Missing or incomplete API information

Failed extractions are logged, and skipped APIs are reported at the end of the execution.
