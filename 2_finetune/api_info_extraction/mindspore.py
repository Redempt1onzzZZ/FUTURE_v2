import requests
from bs4 import BeautifulSoup
import re
import json
import os
from typing import Dict, List, Optional, Tuple

# Base URL
BASE_URL = "https://www.mindspore.cn/docs/en/r2.2/api_python/"

def clean_text(text: str) -> str:
    """Clean text content, remove excess spaces and newlines"""
    return re.sub(r'\s+', ' ', text).strip()

def extract_api_info(api_name: str) -> Optional[Dict]:
    """Scrape detailed information for a specific API"""
    print(f"Scraping API information: {api_name}")
    
    # Fix MindSpore doc URL format:
    # mindspore.ops.abs -> ops/mindspore.ops.abs.html
    parts = api_name.split('.')
    if len(parts) >= 3:  # mindspore.ops.abs
        module = parts[1]  # ops
        url = f"{BASE_URL}{module}/{api_name}.html"
    else:
        url = f"{BASE_URL}{api_name}.html"
    
    # If r2.2 version is not found, try r2.5.0 version
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Cannot access {url}, status code: {response.status_code}")
            # Try r2.5.0 version
            r25_url = url.replace("r2.2", "r2.5.0")
            response = requests.get(r25_url)
            if response.status_code != 200:
                # Try alternative URL format
                alt_url = f"{BASE_URL}api/{api_name}.html"
                response = requests.get(alt_url)
                if response.status_code != 200:
                    print(f"Cannot access alternative URL {alt_url}, status code: {response.status_code}")
                    return None
            else:
                url = r25_url
                print(f"Using r2.5.0 version URL: {url}")
        
        # Save HTML content for debugging
        # with open(f"{api_name.replace('.', '_')}.html", 'w', encoding='utf-8') as f:
        #     f.write(response.text)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract API information
        signature = ""
        description = ""
        parameters = []
        returns = ""
        examples = ""
        
        # Find main content area
        content_div = soup.find('div', class_='section')
        if not content_div:
            content_div = soup.find('div', class_='body')
            if not content_div:
                content_div = soup
        
        # Extract function name and signature
        func_header = soup.find('h1')
        if func_header:
            # Function signature is usually in a code block
            sig_div = soup.find('dl', class_='function') or soup.find('dl', class_='method')
            if sig_div:
                dt_element = sig_div.find('dt')
                if dt_element:
                    code_element = dt_element.find('code', class_='sig-object')
                    if code_element:
                        signature = clean_text(code_element.text)
                    else:
                        signature = clean_text(dt_element.text)
        
        # If signature is still empty, try other methods
        if not signature:
            pre_elements = soup.find_all('pre')
            for pre in pre_elements:
                if api_name.split('.')[-1] in pre.text and '(' in pre.text:
                    signature = clean_text(pre.text)
                    break
        
        # Remove [source] tag from signature
        signature = re.sub(r'\[source\]', '', signature).strip()
        
        # Extract description - optimized version
        # 1. First, try to find the first paragraph after function signature
        description_paragraphs = []
        if sig_div:
            dd_element = sig_div.find('dd')
            if dd_element:
                # Try to find the first paragraph
                p_elements = dd_element.find_all('p')
                if p_elements:
                    description = clean_text(p_elements[0].text)
        
        # 2. If not found above, look for paragraph that appears before Parameters
        if not description:
            main_content = soup.find('div', class_='body') or soup.find('div', class_='document')
            if main_content:
                for p in main_content.find_all('p'):
                    next_element = p.find_next()
                    if next_element and ('Parameters' in next_element.text or 
                                         hasattr(next_element, 'name') and 
                                         next_element.name in ['h2', 'h3'] and 
                                         'Parameters' in next_element.text):
                        description = clean_text(p.text)
                        break
        
        # 3. More generic approach: find first paragraph after h1 heading
        if not description and func_header:
            next_p = func_header.find_next('p')
            if next_p:
                # Extract paragraph content directly as description
                description = clean_text(next_p.text)
        
        # 4. If still no description, try to extract from any paragraph that might contain function description
        if not description:
            p_elements = soup.find_all('p')
            for p in p_elements:
                # First paragraph not containing "Parameters", "Returns", "Raises" etc. might be the description
                if not any(keyword in p.text for keyword in ['Parameters:', 'Returns:', 'Raises:', 'Example']):
                    description = clean_text(p.text)
                    break
        
        # Find field list area (might contain parameters, return values, etc.)
        field_list = soup.find('dl', class_='field-list')
        
        # Extract parameter information
        # 1. Get parameters from field list
        if field_list:
            dt_elements = field_list.find_all('dt')
            for dt in dt_elements:
                if 'Parameters' in dt.text:
                    dd_params = dt.find_next('dd')
                    if dd_params:
                        # Look for parameter list
                        param_list = dd_params.find('ul')
                        if param_list:
                            for li in param_list.find_all('li'):
                                param_text = clean_text(li.text)
                                parameters.append(param_text)
                        else:
                            # Parameters might be directly in dd element
                            param_text = clean_text(dd_params.text)
                            param_lines = param_text.split('\n')
                            for line in param_lines:
                                if line.strip() and ('–' in line or '-' in line):
                                    parameters.append(clean_text(line))
        
        # 2. If not found above, look for parameter paragraph or section
        if not parameters:
            param_section = None
            for section in soup.find_all(['div', 'section']):
                h_tags = section.find_all(['h2', 'h3', 'h4'])
                for h in h_tags:
                    if 'Parameters' in h.text:
                        param_section = section
                        break
                if param_section:
                    break
            
            if not param_section:
                # Try to find parameter table or list
                for div in soup.find_all('div', class_='table-wrapper'):
                    if 'Parameters' in div.text:
                        param_section = div
                        break
            
            if param_section:
                # Parameters might be in a table
                tables = param_section.find_all('table')
                if tables:
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows[1:]:  # Skip header
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                param_name = clean_text(cells[0].text)
                                param_desc = clean_text(cells[1].text)
                                parameters.append(f"{param_name} – {param_desc}")
                else:
                    # Or in a definition list
                    dts = param_section.find_all('dt')
                    for dt in dts:
                        param_name = dt.text.strip()
                        dd = dt.find_next('dd')
                        if dd:
                            param_desc = clean_text(dd.text)
                            parameters.append(f"{param_name} – {param_desc}")
            
            # If still no parameters, try to extract from paragraph text
            if not parameters:
                for p in p_elements:
                    if 'Parameters' in p.text:
                        param_text = p.text.replace('Parameters', '').strip()
                        if param_text:
                            # Try to extract parameters in format "input (Type) - Description"
                            param_matches = re.findall(r'(\w+)\s*\(([^)]+)\)\s*[–-]\s*([^.]+)', param_text)
                            for match in param_matches:
                                param_name, param_type, param_desc = match
                                parameters.append(f"{param_name} ({param_type}) – {param_desc}")
        
        # Extract return value information
        # 1. Get return values from field list
        if field_list:
            dt_elements = field_list.find_all('dt')
            for dt in dt_elements:
                # Extract return value
                if 'Returns' in dt.text and 'Return type' not in dt.text:
                    dd_returns = dt.find_next('dd')
                    if dd_returns:
                        returns = clean_text(dd_returns.text)
        
        # 2. If not obtained from field list, try other areas
        if not returns:
            returns_section = None
            for section in soup.find_all(['div', 'section']):
                h_tags = section.find_all(['h2', 'h3', 'h4'])
                for h in h_tags:
                    if ('Returns' in h.text and 'Parameters' not in h.text):
                        returns_section = section
                        break
                if returns_section:
                    break
            
            if returns_section:
                # Extract return value text
                p_tags = returns_section.find_all('p')
                for p in p_tags:
                    if 'Returns' in p.text and 'Return type' not in p.text and not returns:
                        returns = clean_text(p.text.replace('Returns', '').strip())
        
        # 3. If still not found, try to extract from paragraph text
        if not returns:
            for p in p_elements:
                if p.text.startswith("Returns"):
                    returns_text = p.text.replace("Returns", "").strip()
                    if "Return type" in returns_text:
                        # Separate return value and return type
                        parts = returns_text.split("Return type")
                        returns = clean_text(parts[0])
                    else:
                        returns = clean_text(returns_text)
                    break
        
        # Extract code examples - enhanced version
        # 1. First find Examples section heading
        examples_section = None
        examples_header = None
        
        # Look for Examples heading
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            if 'Examples' in h.text or 'Example' in h.text:
                examples_header = h
                # Found the heading, now find the corresponding section
                examples_section = h.parent
                if not examples_section.find('pre') and not examples_section.find('code'):
                    # If current section has no code blocks, try to find next element
                    next_sibling = h.find_next()
                    if next_sibling:
                        examples_section = next_sibling.parent
                break
        
        # If no clear Examples heading, try other methods
        if not examples_section:
            for section in soup.find_all(['div', 'section']):
                if any(keyword in section.text for keyword in ['Examples', 'Example']):
                    examples_section = section
                    break
        
        # If still not found, try to find paragraph containing 'Example'
        if not examples_section:
            for p in p_elements:
                if p.strong and 'Example' in p.strong.text:
                    examples_section = p.parent
                    break
                elif 'Example' in p.text and not p.find('a'):  # Avoid navigation links
                    examples_section = p.parent
                    break
        
        # Now extract example code
        if examples_section:
            # Try to find code blocks directly
            code_blocks = examples_section.find_all(['pre', 'code'])
            if code_blocks:
                examples = "\n".join([block.text for block in code_blocks])
            else:
                # Look for content right after Examples heading
                example_content = ""
                if examples_header:
                    current = examples_header.next_sibling
                    while current and current.name != 'h1' and current.name != 'h2' and current.name != 'h3':
                        if current.name == 'pre' or current.name == 'code':
                            example_content += current.text + "\n"
                        elif hasattr(current, 'text'):
                            example_content += current.text
                        current = current.next_sibling
                
                if example_content:
                    examples = example_content
                else:
                    # If still not found, get text from the entire section
                    examples = examples_section.text
                    # Remove heading
                    for marker in ['Examples:', 'Example:', 'Examples', 'Example']:
                        if marker in examples:
                            examples = examples[examples.find(marker) + len(marker):].strip()
        
        # Try to extract example code from any pre elements in the page
        if not examples:
            pre_elements = soup.find_all('pre')
            if pre_elements and len(pre_elements) > 1:  # First one is usually the signature
                examples = "\n".join([pre.text for pre in pre_elements[1:]])
        
        # Still no examples, try special rules
        if not examples:
            # Look for text containing '>>>', which typically indicates Python interactive examples
            for p in soup.find_all(['p', 'div']):
                if '>>>' in p.text:
                    examples = p.text
                    break
        
        # Filter out "Supported Platforms" information from examples
        if examples:
            # Handle potential platform support information
            platform_patterns = [
                r'Supported Platforms:[\s\n]*Ascend[\s\n]*GPU[\s\n]*CPU[\s\n]*',
                r'Supported Platforms:.*?(?=>>>|\n\n)',
                r'(?:Ascend|GPU|CPU)(?:\s+(?:Ascend|GPU|CPU))*\s*\n*'
            ]
            
            for pattern in platform_patterns:
                examples = re.sub(pattern, '', examples, flags=re.IGNORECASE)
            
            # Clean any leading empty lines
            examples = re.sub(r'^\s*\n*', '', examples)
            
            # Ensure there is at least some code example content
            if not re.search(r'>>>\s+', examples):
                examples = ""
        
        # Ensure parameters and returns don't get confused
        # 1. Remove potential return information from parameters
        filtered_parameters = []
        for param in parameters:
            if not ('Returns' in param or 'Return type' in param):
                filtered_parameters.append(param)
        parameters = filtered_parameters
        
        # 2. Handle potential return type information in returns
        if returns and 'Return type' in returns:
            parts = returns.split('Return type')
            returns = parts[0].strip()
        
        # Build API information dictionary
        api_info = {
            "name": api_name,
            "signature": signature,
            "description": description,
            "parameters": parameters,
            "returns": returns,
            "examples": examples
        }
        
        # Debug output
        print(f"Extracted for {api_name}:")
        print(f"  Signature: {signature}")
        print(f"  Description: {description}")
        print(f"  Parameters: {parameters}")
        print(f"  Returns: {returns}")
        print(f"  Examples: {examples[:100]}..." if len(examples) > 100 else f"  Examples: {examples}")
        
        return api_info
    
    except Exception as e:
        print(f"Error scraping {api_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def read_api_list(file_path: str) -> List[str]:
    """Read API list from file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def save_api_info(api_info_list: List[Dict], output_file: str) -> None:
    """Save API information to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(api_info_list, f, ensure_ascii=False, indent=2)
    
    print(f"API information saved to {output_file}")

def main():
    # Read API list
    api_list_file = "./2_finetune/api_info_extraction/api_list/mindspore_api.txt"
    output_dir = "./2_finetune/api_info_extraction/api_info"
    output_file = os.path.join(output_dir, "mindspore_api_info.json")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    api_list = read_api_list(api_list_file)
    api_info_list = []
    
    # Test specific API
    # api_list = ["mindspore.ops.numel"]
    
    # Scrape information for each API
    skipped_apis = []
    for api_name in api_list:
        api_info = extract_api_info(api_name)
        if api_info:
            api_info_list.append(api_info)
        else:
            skipped_apis.append(api_name)
    
    # Save scraped information
    save_api_info(api_info_list, output_file)
    
    print(f"Scraped information for {len(api_info_list)} APIs out of {len(api_list)}")
    if skipped_apis:
        print(f"Skipped APIs: {skipped_apis}")

if __name__ == "__main__":
    main() 