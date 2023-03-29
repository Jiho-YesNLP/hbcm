import re
import json

def extract_fields(input_file_path):
    with open(input_file_path, 'r') as file:
        content = file.read()
        
        id_pattern = re.compile(r'"_id"\s*:\s*"([^"]+)"', re.MULTILINE)
        keywords_pattern = re.compile(r'"keywords"\s*:\s*\[(.*?)\]', re.MULTILINE | re.DOTALL)
        fos_pattern = re.compile(r'"fos"\s*:\s*\[(.*?)\]', re.MULTILINE | re.DOTALL)
        
        ids = id_pattern.findall(content)
        keywords = [re.findall(r'"([^"]+)"', k) for k in keywords_pattern.findall(content)]
        fos = [re.findall(r'"([^"]+)"', f) for f in fos_pattern.findall(content)]

        for i, k, f in zip(ids, keywords, fos):
            yield {
                'id': i,
                'keywords': k,
                'fos': f
            }

input_file_path = 'dblpv13.json'

# Extract required fields and save them in the desired format
entries = {}
for paper in extract_fields(input_file_path):
    entry = {'id': paper["id"]}
    
    if paper['keywords']:
        entry['keywords'] = paper['keywords']
    
    if paper['fos']:
        entry['fos'] = paper['fos']
    
    if 'keywords' in entry or 'fos' in entry:
        entries[paper["id"]] = entry

# Dump the entries dictionary to a JSON file
with open('output_entries.json', 'w') as outfile:
    json.dump(entries, outfile, indent=2)


