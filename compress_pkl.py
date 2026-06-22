import joblib
import os

input_file = 'hr_promotion_pipeline1.pkl'
output_file = 'hr_promotion_pipeline1_compressed.pkl'

print(f"Loading {input_file} (Ukuran: {os.path.getsize(input_file) / 1024 / 1024:.2f} MB)...")
model_data = joblib.load(input_file)

print("Compressing and saving... (Ini mungkin memakan waktu beberapa detik)")
joblib.dump(model_data, output_file, compress=3)

print(f"Berhasil! Ukuran baru {output_file}: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
