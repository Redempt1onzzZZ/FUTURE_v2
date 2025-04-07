import requests
from bs4 import BeautifulSoup
import re
import json
import os
from typing import Dict, List, Optional, Tuple

# Base URL
BASE_URL = "https://ml-explore.github.io/mlx/build/html/"

def clean_text(text: str) -> str:
    """Clean text content, remove excess spaces and newlines"""
    return re.sub(r'\s+', ' ', text).strip()

def extract_api_info(api_name: str) -> Optional[Dict]:
    """Scrape detailed information for a specific API"""
    print(f"Scraping API information: {api_name}")
    
    # Format URL based on actual MLX documentation structure
    url = f"{BASE_URL}python/_autosummary/{api_name}.html"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Cannot access {url}, status code: {response.status_code}")
            return None
        
        # Save the HTML content for debugging if needed
        # with open(f"{api_name.replace('.', '_')}.html", 'w', encoding='utf-8') as f:
        #     f.write(response.text)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the function definition section or property definition
        function_def = soup.find('dl', class_='py function')
        property_def = soup.find('dl', class_='py property')
        attribute_def = soup.find('dl', class_='py attribute')
        
        # Use the first available definition
        definition = function_def or property_def or attribute_def
        
        if not definition:
            # Try to find class method or other types
            definition = soup.find(['dl', 'dt'], id=api_name.split('.')[-1])
            
            if not definition:
                # Last attempt: try to find by name in any dl element
                all_dls = soup.find_all('dl')
                for dl in all_dls:
                    if api_name.split('.')[-1] in dl.text:
                        definition = dl
                        break
        
        if not definition:
            print(f"Could not find definition for {api_name}")
            return None
            
        # Extract the signature - this will be different for properties vs functions
        signature = ""
        dt_element = None
        is_property = False
        
        if definition:
            dt_element = definition.find('dt', class_='sig sig-object py')
            
            if not dt_element:
                dt_element = definition.find('dt')
                
            # Check if it's a property
            if property_def or (dt_element and 'property' in dt_element.get('class', [])):
                is_property = True
                
        if dt_element:
            # Get function/property name
            name_element = dt_element.find('span', class_='sig-name descname')
            if name_element:
                name = name_element.text.strip()
            else:
                name = api_name.split('.')[-1]
            
            if is_property:
                # For properties, the signature is simpler
                signature = f"property {name}"
                
                # Check if there's a type annotation
                type_annotation = dt_element.find('span', class_='sig-prename descclassname')
                if type_annotation:
                    signature = f"{signature}: {type_annotation.text.strip()}"
            else:
                # For functions, extract parameters and return type
                params = []
                param_elements = dt_element.find_all('em', class_='sig-param')
                for param in param_elements:
                    param_text = param.get_text().strip()
                    params.append(param_text)
                
                # Get return type
                sig_return = dt_element.find('span', class_='sig-return-icon')
                if sig_return:
                    return_typehint = dt_element.find('span', class_='sig-return-typehint')
                    if return_typehint:
                        return_annotation = f" → {return_typehint.text.strip()}"
                    else:
                        return_annotation = ""
                else:
                    return_annotation = ""
                
                # Build signature
                signature = f"{name}({', '.join(params)}){return_annotation}"
        
        # Extract function/property description
        description = ""
        dd_element = None
        if definition:
            dd_element = definition.find('dd')
            if dd_element:
                # Get only the paragraphs that are not part of a field list
                p_elements = dd_element.find_all('p', recursive=False)
                if not p_elements:
                    # If no paragraphs at the top level, look inside the dd element
                    all_p = dd_element.find_all('p')
                    field_list = dd_element.find('dl', class_='field-list')
                    if field_list:
                        # Exclude paragraphs inside field list
                        field_list_p = field_list.find_all('p')
                        p_elements = [p for p in all_p if p not in field_list_p]
                    else:
                        p_elements = all_p
                
                if p_elements:
                    description = clean_text(p_elements[0].text)
                    # If there's a second paragraph, include it too
                    if len(p_elements) > 1:
                        second_para = clean_text(p_elements[1].text)
                        if second_para:
                            description += " " + second_para
        
        # Extract parameter information
        parameters = []
        if dd_element and not is_property:
            # Try to find Parameters section in field-list
            field_list = dd_element.find('dl', class_='field-list simple')
            if not field_list:
                field_list = dd_element.find('dl', class_='field-list')
                
            if field_list:
                # Look for "Parameters:" label
                dt_elements = field_list.find_all('dt')
                for dt in dt_elements:
                    if 'Parameters' in dt.text:
                        # Find the corresponding dd element
                        dd_params = dt.find_next('dd')
                        if dd_params:
                            # Extract parameters from list items
                            param_list = dd_params.find('ul', class_='simple')
                            if param_list:
                                li_elements = param_list.find_all('li')
                                for li in li_elements:
                                    parameters.append(clean_text(li.text))
                            else:
                                # Some docs have parameters directly in the dd
                                param_text = clean_text(dd_params.text)
                                if param_text:
                                    parameters.append(param_text)
        
        # If parameters are still empty, try to extract from the text (for functions only)
        if not parameters and description and not is_property:
            param_match = re.search(r'([a-zA-Z_]+)\s*\((array|scalar|.*?)\)\s*–\s*(.*?)\.(Returns|$)', description)
            if param_match:
                param_name = param_match.group(1).strip()
                param_type = param_match.group(2).strip()
                param_desc = param_match.group(3).strip()
                parameters.append(f"{param_name} ({param_type}) – {param_desc}")
                
                # Fix the description to remove the parameter part
                description = re.sub(r'\s*[a-zA-Z_]+\s*\((array|scalar|.*?)\)\s*–\s*.*?\.(Returns|$)', r'.\2', description)
                description = description.replace('..', '.')

        # Extract return value information and return type
        returns = ""
        return_type = ""
        if dd_element:
            field_list = dd_element.find('dl', class_='field-list simple')
            if not field_list:
                field_list = dd_element.find('dl', class_='field-list')
                
            if field_list:
                # Find field for Returns
                dt_elements = field_list.find_all(['dt'])
                for dt in dt_elements:
                    if 'Returns' in dt.text and 'Return type' not in dt.text:
                        dd_returns = dt.find_next('dd')
                        if dd_returns:
                            returns = clean_text(dd_returns.text)
                    
                    # Find field for Return type        
                    if 'Return type' in dt.text:
                        dd_return_type = dt.find_next('dd')
                        if dd_return_type:
                            return_type = clean_text(dd_return_type.text)
        
        # If return type is not found, try to extract from the signature
        if not return_type and signature and '→' in signature:
            return_type_match = re.search(r'→\s*([^\s]+)', signature)
            if return_type_match:
                return_type = return_type_match.group(1).strip()
        
        # Build API information dictionary
        api_info = {
            "name": api_name,
            "signature": signature,
            "description": description,
            "parameters": parameters,
            "returns": returns,
            "return_type": return_type
        }
        
        # Debug output for diagnostics
        print(f"Extracted for {api_name}:")
        print(f"  Signature: {signature}")
        print(f"  Description: {description}")
        print(f"  Parameters: {parameters}")
        print(f"  Returns: {returns}")
        print(f"  Return type: {return_type}")
        
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
    api_list_file = "./2_finetune/api_info_extraction/api_list/mlx_api.txt"
    output_dir = "./2_finetune/api_info_extraction/api_info"
    output_file = os.path.join(output_dir, "mlx_api_info.json")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    api_list = read_api_list(api_list_file)
    api_info_list = []
    
    # For testing with specific APIs
    # api_list = ["mlx.core.add", "mlx.core.abs", "mlx.core.matmul"]
    
    # Scrape information for each API
    for api_name in api_list:
        api_info = extract_api_info(api_name)
        if api_info:
            api_info_list.append(api_info)
    
    # Save scraped information
    save_api_info(api_info_list, output_file)
    
    print(f"Scraped information for {len(api_info_list)} APIs")

if __name__ == "__main__":
    main()
