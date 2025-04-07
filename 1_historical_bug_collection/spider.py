import requests
from parsel import Selector
import time
import re
import os

def replace_invalid_chars(name):
    """Replace invalid characters in file/folder names with underscores"""
    return re.sub(r'[\/:*?"<>|]', '_', name)
def preprocess_code(code_text, source):
    """Preprocess code text, add dependencies and clean non-code elements"""
    # Remove Markdown code block markers
    code_text = re.sub(r'```\w*', '', code_text)
    
    # Remove Traceback error information (multi-line pattern)
    traceback_pattern = re.compile(
        r'Traceback \(most recent call last\):\n.*?\n\S+.*?(\n\s+at .*?)*\n',
        re.DOTALL
    )
    code_text = traceback_pattern.sub('', code_text)
    
    # Remove lines containing error descriptions (case insensitive)
    code_lines = []
    for line in code_text.split('\n'):
        line = line.strip()
        # Keep empty lines and normal code lines
        if not line or \
           not (re.search(r'error:?|warning:?|exception:?', line, re.I) and 
                not line.startswith('#') and 
                not line.startswith('"')):
            code_lines.append(line)
    
    # Add necessary dependencies
    if source == 'tensorflow':
        if not re.search(r'import\s+tensorflow\s+as\s+tf', code_text):
            code_lines.insert(0, 'import tensorflow as tf')
    elif source == 'pytorch':
        if not re.search(r'import\s+torch', code_text):
            code_lines.insert(0, 'import torch')
        if not re.search(r'import\s+torch.nn as nn', code_text):
            code_lines.insert(0, 'import torch.nn as nn')
    
    # Remove duplicate empty lines and standardize format
    processed_code = '\n'.join([line for line in code_lines if line.strip() != ''])
    return processed_code
def save_code(issues_id, title, code_text, source):
    """
    Save code snippets from GitHub issues to local files
    
    Args:
        issues_id: GitHub issue ID
        title: Issue title
        code_text: Code snippet content
        source: Source of the code (e.g., 'tensorflow', 'pytorch')
    """

    # Preprocess code
    processed_code = preprocess_code(code_text, source)

    # TODO: Modify the base path according to your environment
    folder_name = issues_id + '_' + title
    folder_name = replace_invalid_chars(folder_name)
    folder_name = f'./Github_Issues_code/{folder_name}'  # Base path for saving files
    
    try:
        # First attempt: Create folder with issue ID and title
        os.makedirs(f'./{folder_name}', exist_ok=True)
        
        code_name = replace_invalid_chars(title)
        i = 1
        code_file_name = f'{folder_name}/{code_name}_{i}.py'
        while os.path.exists(code_file_name):
            i += 1
            code_file_name = f'{folder_name}/{code_name}_{i}.py'
        try:
            with open(code_file_name, 'w', encoding='utf-8-sig') as file:
                file.write(processed_code)
        except:
            # Second attempt: Use only issue ID as folder name if title fails
            code_name = '1'
            folder_name = issues_id
            folder_name = replace_invalid_chars(folder_name)
            folder_name = f'./Github_Issues_code/{folder_name}'
            os.makedirs(f'./{folder_name}', exist_ok=True)
            i = 1
            code_file_name = f'{folder_name}/{code_name}_{i}.py'
            while os.path.exists(code_file_name):
                i += 1
                code_file_name = f'{folder_name}/{code_name}_{i}.py'
            
            with open(code_file_name, 'w', encoding='utf-8-sig') as file:
                file.write(processed_code)
    except:
        # Third attempt: Use truncated folder name if both previous attempts fail
        code_name = '1'
        folder_name = folder_name[:61]  # Truncate folder name to avoid path length issues
        folder_name = replace_invalid_chars(folder_name)
        folder_name = f'./Github_Issues_code/{folder_name}'
        os.makedirs(f'./{folder_name}', exist_ok=True)
        i = 1
        code_file_name = f'{folder_name}/{code_name}_{i}.py'
        while os.path.exists(code_file_name):
            i += 1
            code_file_name = f'{folder_name}/{code_name}_{i}.py'

        with open(code_file_name, 'w', encoding='utf-8-sig') as file:
            file.write(processed_code)
            
    print(f'{code_file_name}')


# HTTP request headers for GitHub
headers = {
    "accept": "text/html, application/xhtml+xml",
    "accept-language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "authority": "github.com",
    "cache-control": "no-cache",
    "dnt": "1",
    "pragma": "no-cache",
    "referer": "https://github.com/tensorflow/tensorflow/issues?page=2&q=is%3Aissue+label%3Atype%3Abug",
    "sec-ch-ua": "\"Chromium\";v=\"122\", \"Not(A:Brand\";v=\"24\", \"Google Chrome\";v=\"122\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "turbo-visit": "true",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def get_issues_list_tensorflow(page):
    """
    Scrape TensorFlow GitHub issues for bug-related code snippets
    
    Args:
        page: Page number to scrape
    """
    time.sleep(3)  # Rate limiting
    cookies = {}

    # Adjust the URL to select the desired issues based on labels
    response = requests.get(
        f"https://github.com/tensorflow/tensorflow/issues?page={page}&q=is%3Aissue+label%3Atype%3Abug",
        headers=headers, 
        cookies=cookies
    )
    
    select = Selector(response.text)
    
    # Extract and process each issue
    for i in select.xpath('//div[@class="flex-auto min-width-0 p-2 pr-3 pr-md-2"]'):
        
        time.sleep(3)
        url = 'https://github.com' + i.xpath('./a/@href').get()
        try:
            title = i.xpath('./a/text()').get().replace("\\", "").strip()
        except:
            title = i.xpath('./a//text()').get().replace("\\", "").strip()
        print(title,url)

        issues_id = url.split('/')[-1]
        
        response1 = requests.get(url,headers=headers)
        select1 = Selector(response1.text)
        
        # Check various sections where code snippets might be present
        if 'Standalone code to reproduce the issue' in response1.text:
            code_text = ''.join(select1.xpath('//h3[text()="Standalone code to reproduce the issue"]/following-sibling::div[1]//pre//text()').getall())
            if code_text != '':
                save_code(issues_id, title, code_text, 'tensorflow')
        if 'Usage example' in response1.text:
            code_text = ''.join(select1.xpath('//h3[text()="Usage example"]/following-sibling::div[1]//pre//text()').getall())
            if code_text != '':
                save_code(issues_id, title, code_text, 'tensorflow')
        if 'Code' in response1.text:
            code_text = ''.join(select1.xpath('//h2[text()="Code"]/following-sibling::div[1]//pre//text()').getall())
            if code_text != '':
                save_code(issues_id, title, code_text, 'tensorflow')
        if 'Standalone code to reproduce the issue' in response1.text:
            code_text = ''.join(select1.xpath('//strong[text()="Standalone code to reproduce the issue"]/../following-sibling::div[1]//pre//text()').getall())
            if code_text != '':
                save_code(issues_id, title, code_text, 'tensorflow')
        if 'Code to reproduce the issue' in response1.text:
            code_text = ''.join(select1.xpath('//strong[text()="Code to reproduce the issue"]/../following-sibling::div[1]//pre//text()').getall())
            if code_text != '':
                save_code(issues_id, title, code_text, 'tensorflow')
        if 'The error can be reproduced using this code (test.py):' in response1.text:
            code_text = ''.join(select1.xpath('//strong[text()="The error can be reproduced using this code (test.py):"]/../following-sibling::div[1]//pre//text()').getall())
            if code_text != '':
                save_code(issues_id, title, code_text, 'tensorflow')
        if 'Source code / logs' in response1.text:
            code_text = ''.join(select1.xpath('//h3[text()="Source code / logs"]/following-sibling::div[1]//pre//text()').getall())
            if code_text != '':
                save_code(issues_id, title, code_text, 'tensorflow')
                
def get_issues_list_pytorch(page):
    """
    Scrape PyTorch GitHub issues for bug-related code snippets
    
    Args:
        page: Page number to scrape
    """
    time.sleep(3)
    
    # Adjust the URL to select the desired issues based on labels
    response = requests.get(
        f"https://github.com/pytorch/pytorch/issues?page={page}&q=is%3Aissue+label%3Abug",
        headers=headers
    )

    select = Selector(response.text)

    for i in select.xpath('//div[@class="flex-auto min-width-0 p-2 pr-3 pr-md-2"]'):
        time.sleep(3)
        url = 'https://github.com' + i.xpath('./a/@href').get()
        title = i.xpath('./a/text()').get().replace("\\", "").strip()
        print(title,url)
        issues_id = url.split('/')[-1]
        response1 = requests.get(url,headers=headers)
        select1 = Selector(response1.text)

        # Similar implementation to TensorFlow function
        if '🐛 Describe the bug' in response1.text:
            code_text = ''.join(select1.xpath('//h3[text()="🐛 Describe the bug"]/following-sibling::div[1]//pre//text()').getall())
            if code_text != '':
                if 'Error:' not in code_text:
                    save_code(issues_id, title, code_text, 'pytorch')
        if 'Code example' in response1.text:
            code_text = ''.join(select1.xpath('//h2[text()="Code example"]/following-sibling::div[1]//pre//text()').getall())
            if code_text != '':
                if 'Error:' not in code_text:
                    save_code(issues_id, title, code_text, 'pytorch')
        if 'The following snippet should reproduce the error:' in response1.text:
            code_text = ''.join(select1.xpath('//p[text()="The following snippet should reproduce the error:"]/following-sibling::div[1]//pre//text()').getall())
            if code_text != '':
                if 'Error:' not in code_text:
                    save_code(issues_id, title, code_text, 'pytorch')
                    
# Main execution
# TODO: Adjust the page ranges based on your needs
for page in range(1,400):
    print(f'{page}')
    get_issues_list_tensorflow(page)

for page in range(1,14):
    print(f'{page}')
    get_issues_list_pytorch(page)