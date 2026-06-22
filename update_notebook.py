import json

file_path = 'smart_employee_promotion.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

new_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "save_model_cell_streamlit",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "import os\n",
    "\n",
    "# Menyimpan model dan objek penting lainnya untuk Streamlit\n",
    "pipeline_objects = {\n",
    "    'model': model,\n",
    "    'le_dict': le_dict,\n",
    "    'feature_cols': feature_cols,\n",
    "    'weights': weights,\n",
    "    'benefit': benefit,\n",
    "    'criteria_names': criteria_names\n",
    "}\n",
    "\n",
    "joblib.dump(pipeline_objects, 'hr_promotion_pipeline.pkl')\n",
    "print('Model dan pipeline objects berhasil disimpan ke hr_promotion_pipeline.pkl!')"
   ]
  }

# Cek apakah kode sudah ditambahkan
already_added = any("hr_promotion_pipeline.pkl" in "".join(c.get("source", [])) for c in notebook["cells"])

if not already_added:
    notebook["cells"].append(new_cell)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print("Berhasil menambahkan cell untuk menyimpan model.")
else:
    print("Cell penyimpan model sudah ada.")
