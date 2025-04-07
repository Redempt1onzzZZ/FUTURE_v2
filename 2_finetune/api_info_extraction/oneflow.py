import requests
from bs4 import BeautifulSoup
import re
import json
import os
from typing import Dict, List, Optional, Tuple

# Base URL
BASE_URL = "https://oneflow.readthedocs.io/en/master/"

def clean_text(text: str) -> str:
    """Clean text content, remove excess spaces and newlines"""
    return re.sub(r'\s+', ' ', text).strip()

def extract_api_info(api_name: str) -> Optional[Dict]:
    """Scrape detailed information for a specific API"""
    print(f"Scraping API information: {api_name}")
    
    # Fix API name (e.g., oneflow.zero should be oneflow.zeros)
    if api_name == "oneflow.zero":
        api_name = "oneflow.zeros"
    
    # Construct the correct URL - OneFlow API documentation URL format is generated/api_name.html
    url = f"{BASE_URL}generated/{api_name}.html"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Cannot access {url}, status code: {response.status_code}")
            # Try other possible URL formats
            # Could be a submodule API
            parts = api_name.split('.')
            if len(parts) > 2:
                submodule = '.'.join(parts[:-1])
                function_name = parts[-1]
                url = f"{BASE_URL}generated/{submodule}.html#{api_name}"
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"Cannot access alternative URL {url}, status code: {response.status_code}")
                    return None
        
        # Save HTML content for debugging
        # with open(f"{api_name.replace('.', '_')}.html", 'w', encoding='utf-8') as f:
        #     f.write(response.text)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract function signature and description
        signature = ""
        description = ""
        parameters = []
        returns = ""
        return_type = ""
        examples = ""
        is_property = False
        
        # Find main content area
        content_div = soup.find('div', class_='section')
        if not content_div:
            content_div = soup.find('div', class_='body')
            if not content_div:
                content_div = soup
        
        # Get function signature
        func_header = soup.find('h1')
        if func_header:
            # Title may contain function name
            func_name = func_header.text.strip()
            
            # Look for function signature code block
            sig_div = soup.find('dl')
            if sig_div:
                dt_element = sig_div.find('dt')
                if dt_element:
                    code_element = dt_element.find('code')
                    if code_element:
                        signature = clean_text(code_element.text)
                    else:
                        signature = clean_text(dt_element.text)
        
        # If signature is empty, try other methods
        if not signature:
            # Try to find from page content
            pre_elements = soup.find_all('pre')
            for pre in pre_elements:
                if api_name.split('.')[-1] in pre.text and '(' in pre.text and ')' in pre.text:
                    signature = clean_text(pre.text)
                    break
        
        # Find function description
        p_elements = soup.find_all('p')
        if p_elements:
            # First paragraph is usually the description
            description = clean_text(p_elements[0].text)
            
            # Check for more description paragraphs
            if len(p_elements) > 1 and not p_elements[1].text.startswith("Parameters"):
                description += " " + clean_text(p_elements[1].text)
        
        # Find field list area (may contain parameters, return values, etc.)
        field_list = soup.find('dl', class_='field-list')
        
        # Find parameter information
        param_section = None
        
        # Find parameters from field list
        if field_list:
            dt_elements = field_list.find_all('dt')
            for dt in dt_elements:
                if 'Parameters' in dt.text:
                    param_section = dt.find_next('dd')
                    
                    if param_section:
                        # Try to find parameter list
                        param_list = param_section.find('ul')
                        if param_list:
                            for li in param_list.find_all('li'):
                                param_text = clean_text(li.text)
                                parameters.append(param_text)
                        else:
                            # If no list, parameters might be directly in paragraph
                            param_text = clean_text(param_section.text)
                            param_items = param_text.split('\n')
                            for item in param_items:
                                if '–' in item or '-' in item:
                                    parameters.append(clean_text(item))
        
        # If not found from field list, try to find parameter section
        if not parameters:
            for section in soup.find_all(['div', 'section']):
                if 'Parameters' in section.text and len(section.find_all('dt')) > 0:
                    param_section = section
                    break
            
            if not param_section:
                # Try to find parameter paragraph
                for p in p_elements:
                    if p.text.startswith("Parameters"):
                        param_section = p.parent
                        break
            
            if param_section:
                # Extract parameter list
                param_dts = param_section.find_all('dt')
                for dt in param_dts:
                    param_name = dt.text.strip()
                    dd = dt.find_next('dd')
                    if dd:
                        param_desc = clean_text(dd.text)
                        parameters.append(f"{param_name} – {param_desc}")
        
        # Find return value information and return type
        # 1. First check field list
        if field_list:
            dt_elements = field_list.find_all('dt')
            for dt in dt_elements:
                # Find return value description
                if 'Returns' in dt.text and 'Return type' not in dt.text:
                    returns_dd = dt.find_next('dd')
                    if returns_dd:
                        returns = clean_text(returns_dd.text)
                
                # Find return type
                if 'Return type' in dt.text:
                    return_type_dd = dt.find_next('dd')
                    if return_type_dd:
                        return_type = clean_text(return_type_dd.text)
        
        # 2. If not found above, look for other areas
        if not returns or not return_type:
            # Look for return value information and type
            returns_section = None
            for section in soup.find_all(['div', 'section']):
                if 'Returns' in section.text and 'Parameters' not in section.text:
                    returns_section = section
                    break
            
            if returns_section:
                # Extract information from return section
                p_elements = returns_section.find_all('p')
                if p_elements:
                    returns_text = clean_text(p_elements[0].text)
                    
                    # Try to separate return value description and type
                    if not returns and 'Returns' in returns_text:
                        returns = returns_text.replace('Returns', '').strip()
                    
                    # Look for Return type
                    for p in p_elements:
                        if 'Return type' in p.text:
                            type_text = p.text.replace('Return type', '').strip()
                            if not return_type:
                                return_type = clean_text(type_text)
                
                # If still no return type, try to find from definition list
                dt_elements = returns_section.find_all('dt')
                for dt in dt_elements:
                    if 'Return type' in dt.text:
                        dd = dt.find_next('dd')
                        if dd and not return_type:
                            return_type = clean_text(dd.text)
        
        # 3. If still not found, try to find from paragraphs
        if not returns:
            for p in p_elements:
                if p.text.startswith("Returns"):
                    returns_text = p.text.replace("Returns", "").strip()
                    if "Return type" in returns_text:
                        # Separate return value and return type
                        parts = returns_text.split("Return type")
                        returns = clean_text(parts[0])
                        if len(parts) > 1 and not return_type:
                            return_type = clean_text(parts[1])
                    else:
                        returns = clean_text(returns_text)
                    break
                elif p.text.startswith("Return type"):
                    if not return_type:
                        return_type = clean_text(p.text.replace("Return type", "")).strip()
        
        # Find code examples
        examples_section = None
        for section in soup.find_all(['div', 'section']):
            if any(keyword in section.text for keyword in ['For example:', 'Examples:', 'Example:']):
                examples_section = section
                break
        
        if not examples_section:
            # Try to find section containing example code
            for p in p_elements:
                if any(keyword in p.text for keyword in ['For example:', 'Examples:', 'Example:']):
                    examples_section = p.parent
                    break
        
        if examples_section:
            # Extract example code blocks
            code_blocks = examples_section.find_all('pre')
            if code_blocks:
                examples = "\n".join([block.text for block in code_blocks])
            else:
                # If no code blocks, try to extract text content as example
                examples = clean_text(examples_section.text)
                for start_marker in ['For example:', 'Examples:', 'Example:']:
                    if start_marker in examples:
                        examples = examples[examples.find(start_marker) + len(start_marker):].strip()
        
        # If description is empty, try to use first line of docstring
        if not description and signature:
            desc_match = re.search(r'\)\s*(.+?)\s*Parameters', soup.text, re.DOTALL)
            if desc_match:
                description = clean_text(desc_match.group(1))
        
        # Ensure parameters, return values and return type don't get confused
        # 1. Remove potential return value and return type from parameters
        filtered_parameters = []
        for param in parameters:
            if not ('Returns' in param or 'Return type' in param):
                filtered_parameters.append(param)
        parameters = filtered_parameters
        
        # 2. Handle return type information in return value
        if returns and 'Return type' in returns:
            parts = returns.split('Return type')
            returns = parts[0].strip()
            if len(parts) > 1 and not return_type:
                return_type = parts[1].strip()
        
        # Build API information dictionary
        api_info = {
            "name": api_name,
            "signature": signature,
            "description": description,
            "parameters": parameters,
            "returns": returns,
            "return_type": return_type,
            "examples": examples
        }
        
        # Debug output
        print(f"Extracted for {api_name}:")
        print(f"  Signature: {signature}")
        print(f"  Description: {description}")
        print(f"  Parameters: {parameters}")
        print(f"  Returns: {returns}")
        print(f"  Return type: {return_type}")
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
    api_list_file = "./2_finetune/api_info_extraction/api_list/oneflow_api.txt"
    output_dir = "./2_finetune/api_info_extraction/api_info"
    output_file = os.path.join(output_dir, "oneflow_api_info.json")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    api_list = read_api_list(api_list_file)
    api_info_list = []
    
    # Test specific API
    # api_list = ["oneflow.bitwise_and"]
    
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