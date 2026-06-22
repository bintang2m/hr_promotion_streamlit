import json

file_path = 'smart_employee_promotion.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "joblib.dump(pipeline_objects, 'hr_promotion_pipeline.pkl')" in line:
                source[i] = line.replace("joblib.dump(pipeline_objects, 'hr_promotion_pipeline.pkl')", "joblib.dump(pipeline_objects, 'hr_promotion_pipeline.pkl', compress=3)")

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Berhasil menambahkan kompresi pada notebook.")
