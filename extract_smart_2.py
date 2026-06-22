import json

file_path = 'smart_employee_promotion_2.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

with open('smart_2.py', 'w', encoding='utf-8') as f:
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            f.write("".join(cell['source']) + "\n")

print("Done")
