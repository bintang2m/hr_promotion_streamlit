import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt

st.set_page_config(page_title="AURA - HR Promotion Simulator", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS UNTUK TAMPILAN UNIK ---
st.markdown("""
    <style>
    .main-header {
        font-size: 38px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #3498db, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sub-header {
        font-size: 16px;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 30px;
        font-weight: 500;
        letter-spacing: 1px;
    }
    .metric-card {
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #3498db;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">AURA: HR Promotion Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Utility for Resource Allocation | Model Skenario What-If</div>', unsafe_allow_html=True)

@st.cache_resource
def get_model_pipeline(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model_path = 'hr_promotion_pipeline1_compressed.pkl'
pipeline_objects = get_model_pipeline(model_path)

if pipeline_objects is None:
    st.error("Sistem Belum Siap: File hr_promotion_pipeline1_compressed.pkl tidak ditemukan.")
    st.stop()

model = pipeline_objects['model']
le_dict = pipeline_objects['le_dict']
feature_cols = pipeline_objects['feature_cols']
weights = pipeline_objects['weights']
benefit = pipeline_objects['benefit']
criteria_names = pipeline_objects['criteria_names']

def topsis(matrix, weights, benefit_criteria):
    norm = matrix / np.sqrt((matrix**2).sum(axis=0) + 1e-8)
    weighted = norm * weights
    ideal_pos = np.where(benefit_criteria, weighted.max(axis=0), weighted.min(axis=0))
    ideal_neg = np.where(benefit_criteria, weighted.min(axis=0), weighted.max(axis=0))
    dist_pos = np.sqrt(((weighted - ideal_pos)**2).sum(axis=1))
    dist_neg = np.sqrt(((weighted - ideal_neg)**2).sum(axis=1))
    score = dist_neg / (dist_pos + dist_neg + 1e-8)
    return score

# --- SIDEBAR: Variabel Kontrol ---
st.sidebar.markdown("### Profil Karyawan (Baseline)")
dept = st.sidebar.selectbox("Department", list(le_dict['department'].classes_))
region = st.sidebar.selectbox("Region", list(le_dict['region'].classes_))
edu = st.sidebar.selectbox("Education", ["Bachelor's", "Master's & above", "Below Secondary"])
gender = st.sidebar.selectbox("Gender", list(le_dict['gender'].classes_))
rec_channel = st.sidebar.selectbox("Recruitment Channel", list(le_dict['recruitment_channel'].classes_))

age = st.sidebar.number_input("Usia", min_value=20, max_value=60, value=30)
length_of_service = st.sidebar.number_input("Masa Kerja (Tahun)", min_value=1, max_value=40, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Tuas Kebijakan (Intervensi)")
st.sidebar.caption("Sesuaikan parameter ini untuk mensimulasikan dampak pengembangan sumber daya manusia.")

no_of_trainings = st.sidebar.slider("Jumlah Pelatihan Ekstra", 1, 10, 1)
previous_year_rating = st.sidebar.slider("Target Rating Performa", 1.0, 5.0, 3.0, 1.0)
avg_training_score = st.sidebar.slider("Target Skor Pelatihan", 39, 99, 50)
awards_won = st.sidebar.selectbox("Mendapat Penghargaan Prestasi?", ["Tidak", "Ya"])
awards_won_val = 1 if awards_won == "Ya" else 0

def preprocess_input(dept, region, edu, gender, rec_channel, no_of_trainings, age, prev_rating, service, awards, training_score):
    edu_map = {'Below Secondary': 1, "Bachelor's": 2, "Master's & above": 3}
    
    data = {
        'department_enc': le_dict['department'].transform([dept])[0],
        'region_enc': le_dict['region'].transform([region])[0],
        'education_ord': edu_map[edu],
        'gender_enc': le_dict['gender'].transform([gender])[0],
        'recruitment_channel_enc': le_dict['recruitment_channel'].transform([rec_channel])[0],
        'no_of_trainings': no_of_trainings,
        'age': age,
        'previous_year_rating': prev_rating,
        'length_of_service': service,
        'awards_won?': awards,
        'avg_training_score': training_score
    }
    
    data['rating_x_training'] = data['previous_year_rating'] * data['avg_training_score']
    data['total_score'] = data['avg_training_score'] * data['previous_year_rating'] + data['awards_won?'] * 50
    data['experience_rating'] = data['length_of_service'] * data['previous_year_rating']
    data['training_per_service'] = data['avg_training_score'] / (data['length_of_service'] + 1)
    data['high_performer'] = int((data['previous_year_rating'] >= 4) and (data['avg_training_score'] >= 75))
    data['awards_x_rating'] = data['awards_won?'] * data['previous_year_rating']
    data['rating_squared'] = data['previous_year_rating'] ** 2
    data['training_high'] = int(data['avg_training_score'] >= 80)
    data['achievement_score'] = data['awards_won?'] + data['training_high'] + int(data['previous_year_rating'] >= 4)
    
    data['dept_promo_rate'] = 0.08
    data['dept_avg_training'] = 65.0
    data['dept_avg_rating'] = 3.5
    data['training_vs_dept'] = data['avg_training_score'] - data['dept_avg_training']
    data['rating_vs_dept'] = data['previous_year_rating'] - data['dept_avg_rating']
    
    df = pd.DataFrame([data])
    return df[feature_cols]

# Baseline logic
baseline_input = preprocess_input(dept, region, edu, gender, rec_channel, 1, age, 3.0, length_of_service, 0, 50)
baseline_prob = model.predict_proba(baseline_input)[0][1]

# Intervensi logic
interv_input = preprocess_input(dept, region, edu, gender, rec_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won_val, avg_training_score)
interv_prob = model.predict_proba(interv_input)[0][1]

# Calculate TOPSIS
matrix = np.array([
    [baseline_prob, length_of_service, 0, 50, 3.0],
    [interv_prob, length_of_service, awards_won_val, avg_training_score, previous_year_rating]
]).astype(float)
t_scores = topsis(matrix, weights, benefit)
base_topsis, interv_topsis = t_scores[0], t_scores[1]

delta_prob = interv_prob - baseline_prob
delta_topsis = interv_topsis - base_topsis

# --- UI: PANEL HASIL ---
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Probabilitas Kelulusan Promosi", f"{interv_prob*100:.1f}%", f"{delta_prob*100:.1f}% dari Baseline")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Indeks Prioritas TOPSIS", f"{interv_topsis:.3f}", f"{delta_topsis:.3f} dari Baseline")
    st.markdown('</div>', unsafe_allow_html=True)

if delta_prob > 0.05:
    st.success(f"Analisis Sistem: Keputusan intervensi ini secara substansial MENINGKATKAN peluang promosi karyawan sebesar {delta_prob*100:.1f}%.")
elif delta_prob < -0.05:
    st.warning(f"Analisis Sistem: Keputusan intervensi ini justru MENURUNKAN peluang promosi sebesar {abs(delta_prob)*100:.1f}%. Harap evaluasi ulang.")
else:
    st.info("Analisis Sistem: Modifikasi parameter yang Anda lakukan tidak mengubah profil kandidat secara signifikan.")

st.divider()

# --- UI: VISUALISASI GRAFIK ALTAIR ---
st.markdown("### Perbandingan Komprehensif Skenario")

# Menyiapkan data untuk Altair
data_chart = pd.DataFrame({
    'Skenario': ['Baseline', 'Intervensi'],
    'Probabilitas (%)': [baseline_prob * 100, interv_prob * 100],
    'TOPSIS Score': [base_topsis, interv_topsis]
})

chart1 = alt.Chart(data_chart).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
    x=alt.X('Skenario:N', axis=alt.Axis(labelAngle=0, title='')),
    y=alt.Y('Probabilitas (%):Q', scale=alt.Scale(domain=[0, 100])),
    color=alt.Color('Skenario:N', scale=alt.Scale(range=['#dfe6e9', '#0984e3']), legend=None),
    tooltip=['Skenario', 'Probabilitas (%)']
).properties(title='Perbandingan Probabilitas Model ML', height=350)

text1 = chart1.mark_text(
    align='center',
    baseline='bottom',
    dy=-5,
    fontSize=14,
    fontWeight='bold'
).encode(
    text=alt.Text('Probabilitas (%):Q', format='.1f')
)

chart2 = alt.Chart(data_chart).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
    x=alt.X('Skenario:N', axis=alt.Axis(labelAngle=0, title='')),
    y=alt.Y('TOPSIS Score:Q', scale=alt.Scale(domain=[0, 1.0])),
    color=alt.Color('Skenario:N', scale=alt.Scale(range=['#dfe6e9', '#6c5ce7']), legend=None),
    tooltip=['Skenario', 'TOPSIS Score']
).properties(title='Perbandingan Peringkat Multi-Kriteria (TOPSIS)', height=350)

text2 = chart2.mark_text(
    align='center',
    baseline='bottom',
    dy=-5,
    fontSize=14,
    fontWeight='bold'
).encode(
    text=alt.Text('TOPSIS Score:Q', format='.3f')
)

col_chart1, col_chart2 = st.columns(2)
with col_chart1:
    st.altair_chart(chart1 + text1, use_container_width=True)
with col_chart2:
    st.altair_chart(chart2 + text2, use_container_width=True)


