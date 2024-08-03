"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json

def read_original_data(filepath):
    """
    Reads the original JSON data from a file.
    
    Args:
    filepath (str): Path to the original JSON file.
    
    Returns:
    data (list): Original data as a list of dictionaries.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def convert_to_mlcroissant(data):
    """
    Converts the original dataset to the mlcroissant structured format.
    
    Args:
    data (list): Original data as a list of dictionaries.
    
    Returns:
    dict: Dataset in mlcroissant format.
    """
    mlcroissant_dataset = {
        "@type": "sc:Dataset",
        "name": "DisCRn",
        "description": "A Dataset for Discriminayoty Cross-Modal Reasoning",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": "TBD",
        "distribution": [
            {
            "@type": "cr:FileObject",
            "@id": "discrn_balanced.json",
            "name": "discrn_balanced.json",
            "contentUrl": "data/discrn_balanced.json",
            "encodingFormat": "text/json",
            "sha256": "ae0444ed6d6929757477fd59496979088832acb91abe42facb1feeb73e0b886a"
            }
        ],
        "recordSet": [{
            "@type": "cr:RecordSet",
            "name": "Example Entries",
            "description": "Each record represents a cross-modal reasoning question with answer and multimodal options.",
            "field": []
        }]
    }
    
    fields = set()
    for item in data:
        for key in item:
            if key not in fields:
                fields.add(key)
                field_entry = {
                    "@type": "cr:Field",
                    "name": key,
                    "description": f"Data related to {key}.",
                    "dataType": "sc:Text" if isinstance(item[key], str) else "sc:ItemList"
                }
                mlcroissant_dataset["recordSet"][0]["field"].append(field_entry)
    
    return mlcroissant_dataset

def write_mlcroissant_data(data, output_filepath):
    """
    Writes the converted mlcroissant data to a JSON file.
    
    Args:
    data (dict): Data in mlcroissant format.
    output_filepath (str): Path to the output file.
    """
    with open(output_filepath, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
if __name__ == '__main__':
    original_data = read_original_data('../data/discrn_balanced.json')  # Adjust the filename as necessary
    mlcroissant_data = convert_to_mlcroissant(original_data)
    write_mlcroissant_data(mlcroissant_data, '../data/discrn_balanced_mlcroissant.json')  # Adjust the filename as necessary
