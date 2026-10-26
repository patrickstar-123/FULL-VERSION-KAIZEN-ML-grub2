import pandas as pd
import numpy as np

print("=== Phase 2: Memulai Analisis Monitoring Kustom ===")

# 1. Membaca data train.csv asli Anda
df = pd.read_csv('train.csv')

# 2. Simulasi: Membagi data jadi data Training Lama (Reference) dan Data Baru (Current)
reference_data = df.sample(frac=0.5, random_state=42)
current_data = df.drop(reference_data.index)

# 3. Membuat Fungsi Kalkulasi Data Drift Sederhana
def hitung_drift(ref, curr, kolom):
    # Memastikan tipe data bukan string/object sebelum menghitung rata-rata
    if ref[kolom].dtype == 'object' or pd.api.types.is_string_dtype(ref[kolom]):
        ref_dist = ref[kolom].value_counts(normalize=True)
        curr_dist = curr[kolom].value_counts(normalize=True)
        diff = (ref_dist - curr_dist).abs().sum()
        status = "Terjadi Pergeseran (Drift)" if diff > 0.2 else "Stabil (Aman)"
        return f"<tr><td><b>{kolom}</b></td><td>Kategorikal/Teks</td><td>{diff:.4f}</td><td>{status}</td></tr>"
    else:
        # Konversi paksa ke angka untuk menghindari error reduction string
        ref_numeric = pd.to_numeric(ref[kolom], errors='coerce')
        curr_numeric = pd.to_numeric(curr[kolom], errors='coerce')
        
        ref_mean = ref_numeric.mean()
        curr_mean = curr_numeric.mean()
        diff = abs(ref_mean - curr_mean) / (ref_mean + 1e-5)
        status = "Terjadi Pergeseran (Drift)" if diff > 0.1 else "Stabil (Aman)"
        return f"<tr><td><b>{kolom}</b></td><td>Numerik</td><td>{diff:.4f}</td><td>{status}</td></tr>"

# 4. Menyusun Struktur Dashboard HTML
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Evidently AI Alternative - Data Drift Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f9; }
        h1 { color: #1e293b; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4f46e5; color: white; }
        tr:hover { background-color: #f1f5f9; }
    </style>
</head>
<body>
    <h1>📊 Laporan Monitoring Data & Target Drift (Evidently AI Emulator)</h1>
    <p>Hasil analisis perbandingan data training baseline (Reference) dengan data operasional baru (Current).</p>
    <table>
        <tr>
            <th>Nama Fitur/Kolom</th>
            <th>Tipe Data</th>
            <th>Skor Pergeseran (Drift Score)</th>
            <th>Status Keandalan Model</th>
        </tr>
"""

kolom_uji = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
for col in kolom_uji:
    html_content += hitung_drift(reference_data, current_data, col)

html_content += """
    </table>
</body>
</html>
"""

# 5. Menyimpan Laporan Dashboard HTML
with open('evidently_report.html', 'w',encoding='utf-8') as f:
    f.write(html_content)

print("=== Selesai! Dashboard berhasil dibuat: evidently_report.html ===")
