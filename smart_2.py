import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import shap

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')
print("All libraries loaded successfully!")
train = pd.read_csv('../materi-5/train.csv')
test = pd.read_csv('../materi-5/test.csv')
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
train.head()
print("=== Info Dataset ===")
print(train.dtypes)
print("\n=== Missing Values ===")
print(train.isnull().sum())
print("\n=== Statistik Deskriptif ===")
train.describe()
print("=== Distribusi Target (is_promoted) ===")
print(train['is_promoted'].value_counts())
print(f"\nPersentase Promoted: {train['is_promoted'].mean()*100:.2f}%")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train['is_promoted'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue','coral'])
axes[0].set_title('Distribusi Target'); axes[0].set_xticklabels(['Not Promoted','Promoted'], rotation=0)
for col in ['department','education','gender']:
    print(f"\n{col}: {train[col].value_counts().to_dict()}")
plt.tight_layout(); plt.show()
# Handle missing values
print("Missing before:")
print(train[['education','previous_year_rating']].isnull().sum())

# Impute education with mode, rating with median
train['education'] = train['education'].fillna(train['education'].mode()[0])
train['previous_year_rating'] = train['previous_year_rating'].fillna(train['previous_year_rating'].median())
test['education'] = test['education'].fillna(test['education'].mode()[0])
test['previous_year_rating'] = test['previous_year_rating'].fillna(test['previous_year_rating'].median())

print("\nMissing after:")
print(train[['education','previous_year_rating']].isnull().sum())
# Label Encoding untuk variabel kategorikal
le_dict = {}
cat_cols = ['department', 'region', 'gender', 'recruitment_channel']
df_all = pd.concat([train, test], axis=0, ignore_index=True)

for col in cat_cols:
    le = LabelEncoder()
    le.fit(df_all[col].astype(str))
    train[col + '_enc'] = le.transform(train[col].astype(str))
    test[col + '_enc'] = le.transform(test[col].astype(str))
    le_dict[col] = le
    print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Ordinal Encoding untuk education
edu_map = {'Below Secondary': 1, "Bachelor's": 2, "Master's & above": 3}
train['education_ord'] = train['education'].map(edu_map)
test['education_ord'] = test['education'].map(edu_map)
print(f"\neducation (ordinal): {edu_map}")

# === Group-level Statistics (sangat powerful untuk RF) ===
# Hitung rata-rata promosi per departemen dari training set saja
dept_promo_rate = train.groupby('department')['is_promoted'].mean().to_dict()
dept_avg_training = train.groupby('department')['avg_training_score'].mean().to_dict()
dept_avg_rating = train.groupby('department')['previous_year_rating'].mean().to_dict()

for df in [train, test]:
    df['dept_promo_rate'] = df['department'].map(dept_promo_rate)
    df['dept_avg_training'] = df['department'].map(dept_avg_training)
    df['dept_avg_rating'] = df['department'].map(dept_avg_rating)

print("\nDept promo rates:", {k: round(v, 3) for k, v in dept_promo_rate.items()})

# === Interaction & Polynomial Features ===
for df in [train, test]:
    df['rating_x_training'] = df['previous_year_rating'] * df['avg_training_score']
    df['total_score'] = df['avg_training_score'] * df['previous_year_rating'] + df['awards_won?'] * 50
    df['experience_rating'] = df['length_of_service'] * df['previous_year_rating']
    df['training_per_service'] = df['avg_training_score'] / (df['length_of_service'] + 1)
    df['high_performer'] = ((df['previous_year_rating'] >= 4) & (df['avg_training_score'] >= 75)).astype(int)
    df['awards_x_rating'] = df['awards_won?'] * df['previous_year_rating']
    df['rating_squared'] = df['previous_year_rating'] ** 2
    df['training_high'] = (df['avg_training_score'] >= 80).astype(int)
    df['achievement_score'] = df['awards_won?'] + df['training_high'] + (df['previous_year_rating'] >= 4).astype(int)
    # Deviasi dari rata-rata departemen
    df['training_vs_dept'] = df['avg_training_score'] - df['dept_avg_training']
    df['rating_vs_dept'] = df['previous_year_rating'] - df['dept_avg_rating']

print("\nAll features engineered successfully!")

feature_cols = ['department_enc', 'region_enc', 'education_ord', 'gender_enc',
                'recruitment_channel_enc', 'no_of_trainings', 'age',
                'previous_year_rating', 'length_of_service', 'awards_won?',
                'avg_training_score',
                'dept_promo_rate', 'dept_avg_training', 'dept_avg_rating',
                'rating_x_training', 'total_score', 'experience_rating',
                'training_per_service', 'high_performer', 'awards_x_rating',
                'rating_squared', 'training_high', 'achievement_score',
                'training_vs_dept', 'rating_vs_dept']
print(f"\nTotal Features: {len(feature_cols)}")
X = train[feature_cols]
y = train['is_promoted']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")

# SMOTE ringan (ratio 0.3) + class_weight='balanced' pada model
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"After SMOTE:  {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")
print("\nNote: SMOTE ratio=0.3 (ringan) + class_weight='balanced' di model RF")
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, make_scorer

# === Hyperparameter Tuning ===
param_dist = {
    'n_estimators': [800, 1000],
    'max_depth': [10, 12, 15, 18, 20],
    'min_samples_split': [5, 10, 15, 20],
    'min_samples_leaf': [2, 4, 6, 8],
    'max_features': ['sqrt', 'log2', 0.3],
    'criterion': ['gini', 'entropy'],
}

base_model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

f1_scorer = make_scorer(f1_score)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring=f1_scorer,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("=== Running RandomizedSearchCV (50 iterations, 5-fold CV) ===")
search.fit(X_train_sm, y_train_sm)

print(f"\nBest Parameters: {search.best_params_}")
print(f"Best CV F1-Score: {search.best_score_:.4f}")

model = search.best_estimator_
y_proba = model.predict_proba(X_val)[:, 1]

# === Optimasi Threshold (F1-Score) - fine-grained search ===
best_f1 = 0
best_threshold = 0.5
for t in np.arange(0.10, 0.70, 0.005):
    y_t = (y_proba >= t).astype(int)
    f1_t = f1_score(y_val, y_t)
    if f1_t > best_f1:
        best_f1 = f1_t
        best_threshold = t

print(f"\nOptimal Threshold: {best_threshold:.4f} (default: 0.50)")
print(f"Best F1-Score: {best_f1:.4f}")

y_pred = (y_proba >= best_threshold).astype(int)

print("\n=== Model Evaluation (Optimized) ===")
print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred):.4f}")
print(f"Recall:    {recall_score(y_val, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_val, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_val, y_proba):.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_val, y_pred, target_names=['Not Promoted','Promoted']))
# Confusion Matrix & ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not','Promoted'], yticklabels=['Not','Promoted'])
axes[0].set_title('Confusion Matrix'); axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

fpr, tpr, _ = roc_curve(y_val, y_proba)
axes[1].plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {roc_auc_score(y_val, y_proba):.4f}')
axes[1].plot([0,1],[0,1],'r--'); axes[1].set_title('ROC Curve')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].legend()
plt.tight_layout(); plt.savefig('model_eval.png', dpi=150, bbox_inches='tight'); plt.show()


X_val_sample = X_val.sample(n=500, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val_sample)

plt.figure(figsize=(10, 6))
shap_vals_plot = shap_values[:, :, 1] if type(shap_values).__name__ == 'ndarray' and len(shap_values.shape) == 3 else shap_values[1]
shap.summary_plot(shap_vals_plot, X_val_sample, feature_names=feature_cols, show=False)
plt.title('SHAP Summary Plot - Feature Importance')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
# SHAP Bar Plot
plt.figure(figsize=(10, 5))
shap_vals_plot = shap_values[:, :, 1] if type(shap_values).__name__ == 'ndarray' and len(shap_values.shape) == 3 else shap_values[1]
shap.summary_plot(shap_vals_plot,  X_val_sample, feature_names=feature_cols, plot_type='bar', show=False)
plt.title('SHAP Feature Importance (Bar)')
plt.tight_layout(); plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight'); plt.show()
test_proba = model.predict_proba(test[feature_cols])[:, 1]
test['promo_probability'] = test_proba
test['predicted_promoted'] = (test_proba >= 0.5).astype(int)
print(f"Jumlah diprediksi promoted: {test['predicted_promoted'].sum()} dari {len(test)}")
test[['employee_id','department','promo_probability','predicted_promoted']].head(10)
candidates = test.nlargest(20, 'promo_probability').copy()

criteria_cols = ['promo_probability', 'length_of_service', 'awards_won?',
                 'avg_training_score', 'previous_year_rating']
criteria_names = ['Prob_Promosi', 'Masa_Kerja', 'Awards', 'Skor_Latih', 'Rating']

decision_matrix = candidates[criteria_cols].copy()
decision_matrix.columns = criteria_names
decision_matrix.index = candidates['employee_id']
print("=== Decision Matrix (Raw) ===")
decision_matrix
ahp_matrix = np.array([
    [1,   3,   2,   2,   3],   # Prob_Promosi
    [1/3, 1,   1/2, 1/2, 1],   # Masa_Kerja
    [1/2, 2,   1,   1,   2],   # Awards
    [1/2, 2,   1,   1,   2],   # Skor_Latih
    [1/3, 1,   1/2, 1/2, 1],   # Rating
])

# Hitung bobot AHP (geometric mean method)
geo_mean = np.prod(ahp_matrix, axis=1) ** (1/ahp_matrix.shape[1])
weights = geo_mean / geo_mean.sum()

# Consistency check
col_sum = ahp_matrix.sum(axis=0)
norm_matrix = ahp_matrix / col_sum
priority = norm_matrix.mean(axis=1)
lambda_max = (ahp_matrix @ priority / priority).mean()
CI = (lambda_max - len(weights)) / (len(weights) - 1)
RI = 1.12 
CR = CI / RI

print("=== AHP Weights ===")
for n, w in zip(criteria_names, weights):
    print(f"  {n:15s}: {w:.4f}")
print(f"\nConsistency Ratio (CR): {CR:.4f}")
print(f"CR < 0.1? {'YES - Consistent!' if CR < 0.1 else 'NO - Inconsistent!'}")
def topsis(matrix, weights, benefit_criteria):
    # Step 1: Normalize
    norm = matrix / np.sqrt((matrix**2).sum(axis=0))
    # Step 2: Weighted normalized
    weighted = norm * weights
    # Step 3: Ideal solutions
    ideal_pos = np.where(benefit_criteria, weighted.max(axis=0), weighted.min(axis=0))
    ideal_neg = np.where(benefit_criteria, weighted.min(axis=0), weighted.max(axis=0))
    # Step 4: Distance
    dist_pos = np.sqrt(((weighted - ideal_pos)**2).sum(axis=1))
    dist_neg = np.sqrt(((weighted - ideal_neg)**2).sum(axis=1))
    # Step 5: Closeness coefficient
    score = dist_neg / (dist_pos + dist_neg)
    return score

# All criteria are benefit (higher is better)
benefit = np.array([True, True, True, True, True])
dm_values = decision_matrix.values.astype(float)

topsis_scores = topsis(dm_values, weights, benefit)
decision_matrix['TOPSIS_Score'] = topsis_scores
decision_matrix['Rank'] = decision_matrix['TOPSIS_Score'].rank(ascending=False).astype(int)
decision_matrix_sorted = decision_matrix.sort_values('Rank')

print("=== Ranking Kandidat Promosi (TOPSIS) ===")
decision_matrix_sorted
# Visualisasi Ranking
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(decision_matrix_sorted)))[::-1]
bars = ax.barh(range(len(decision_matrix_sorted)),
               decision_matrix_sorted['TOPSIS_Score'],
               color=colors)
ax.set_yticks(range(len(decision_matrix_sorted)))
ax.set_yticklabels([f"EMP-{eid}" for eid in decision_matrix_sorted.index])
ax.set_xlabel('TOPSIS Score')
ax.set_title('Ranking Kandidat Promosi (Top 20)')
ax.invert_yaxis()
for i, (v, r) in enumerate(zip(decision_matrix_sorted['TOPSIS_Score'], decision_matrix_sorted['Rank'])):
    ax.text(v + 0.005, i, f'#{r} ({v:.3f})', va='center', fontsize=9)
plt.tight_layout(); plt.savefig('promotion_ranking.png', dpi=150, bbox_inches='tight'); plt.show()
scaler = MinMaxScaler()
dm_normalized = pd.DataFrame(
    scaler.fit_transform(decision_matrix[criteria_names]),
    columns=[f"{c}_norm" for c in criteria_names],
    index=decision_matrix.index
)

comparison = pd.concat([decision_matrix[criteria_names], dm_normalized], axis=1)
print("=== Perbandingan Sebelum & Sesudah Normalisasi ===")
comparison.head(10)
# Buat 5 kandidat sintetis untuk simulasi
synthetic = pd.DataFrame({
    'Employee': ['Senior_LowPerf', 'Junior_HighPerf', 'Mid_Balanced', 'Senior_HighPerf', 'Junior_LowPerf'],
    'Prob_Promosi': [0.20, 0.90, 0.55, 0.85, 0.15],
    'Masa_Kerja':  [25,   3,    10,   20,   2],
    'Awards':      [0,    1,    1,    1,    0],
    'Skor_Latih':  [45,   88,   65,   90,   40],
    'Rating':      [2.0,  5.0,  3.5,  4.5,  2.0]
}).set_index('Employee')

print("=== Profil Kandidat Sintetis ===")
print(synthetic)

syn_vals = synthetic.values.astype(float)
syn_scores = topsis(syn_vals, weights, benefit)
synthetic['TOPSIS_Score'] = syn_scores
synthetic['Rank'] = synthetic['TOPSIS_Score'].rank(ascending=False).astype(int)
print("\n=== Ranking Baseline ===")
print(synthetic.sort_values('Rank'))
# What-If: Ubah avg_training_score Senior_LowPerf dari 45 -> 80
scenarios = {}
syn_base = synthetic[criteria_names].copy()

# Scenario 1: Senior improves training score
syn_s1 = syn_base.copy()
syn_s1.loc['Senior_LowPerf', 'Skor_Latih'] = 80
s1_scores = topsis(syn_s1.values.astype(float), weights, benefit)
scenarios['S1: Senior +Training'] = dict(zip(syn_base.index, s1_scores))

# Scenario 2: Junior wins award
syn_s2 = syn_base.copy()
syn_s2.loc['Junior_LowPerf', 'Awards'] = 1
syn_s2.loc['Junior_LowPerf', 'Skor_Latih'] = 75
s2_scores = topsis(syn_s2.values.astype(float), weights, benefit)
scenarios['S2: Junior +Award+Train'] = dict(zip(syn_base.index, s2_scores))

# Scenario 3: Heavier weight on seniority (masa kerja)
weights_senior = np.array([0.25, 0.35, 0.10, 0.15, 0.15])
s3_scores = topsis(syn_base.values.astype(float), weights_senior, benefit)
scenarios['S3: Weight Seniority'] = dict(zip(syn_base.index, s3_scores))

# Compare
baseline = dict(zip(syn_base.index, topsis(syn_base.values.astype(float), weights, benefit)))
result_df = pd.DataFrame({'Baseline': baseline, **scenarios})
result_df = result_df.round(4)
print("=== What-If Comparison ===")
print(result_df)
print("\nKesimpulan: Karyawan senior dengan performa rendah TIDAK otomatis masuk teratas.")
print("Sistem hybrid ML-MCDM memastikan keputusan berbasis multi-kriteria, bukan satu faktor saja.")
# Visualisasi What-If
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(result_df))
w = 0.2
for i, col in enumerate(result_df.columns):
    ax.bar(x + i*w, result_df[col], w, label=col)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(result_df.index, rotation=15)
ax.set_ylabel('TOPSIS Score')
ax.set_title('What-If Analysis: Perubahan Ranking Berdasarkan Skenario')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(); plt.savefig('what_if_analysis.png', dpi=150, bbox_inches='tight'); plt.show()
# def predict_and_rank(employee_profiles, model, le_dict, feature_cols, weights, benefit):
#     """Simulasi dashboard: input profil -> prediksi -> ranking"""
#     df = pd.DataFrame(employee_profiles)
    
#     # Encode categorical
#     for col in ['department', 'region', 'education', 'gender', 'recruitment_channel']:
#         df[col + '_enc'] = le_dict[col].transform(df[col].astype(str))
    
#     # Predict probability
#     proba = model.predict_proba(df[feature_cols])[:, 1]
#     df['promo_probability'] = proba
    
#     # Build decision matrix
#     dm = df[['promo_probability','length_of_service','awards_won?',
#              'avg_training_score','previous_year_rating']].values.astype(float)
#     scores = topsis(dm, weights, benefit)
#     df['TOPSIS_Score'] = scores
#     df['Rank'] = pd.Series(scores).rank(ascending=False).astype(int).values
    
#     return df.sort_values('Rank')

def predict_and_rank(employee_profiles, model, le_dict, feature_cols, weights, benefit):
    """Simulasi dashboard: input profil -> prediksi -> ranking"""
    df = pd.DataFrame(employee_profiles)
    
    # Encode categorical (department, region, gender, recruitment_channel)
    for col in ['department', 'region', 'gender', 'recruitment_channel']:
        if col in le_dict:  # Pastikan key ada
            df[col + '_enc'] = le_dict[col].transform(df[col].astype(str))
        else:
            # Fallback: jika tidak ada di le_dict, buat encoding sederhana
            df[col + '_enc'] = df[col].astype('category').cat.codes
    
    # Ordinal encoding untuk education (manual mapping)
    edu_map = {'Below Secondary': 1, "Bachelor's": 2, "Master's & above": 3}
    df['education_ord'] = df['education'].map(edu_map)
    
    # Handle missing values (jika ada education yang tidak dikenal)
    df['education_ord'] = df['education_ord'].fillna(2)  # default ke Bachelor's
    
    # Pastikan semua feature_cols tersedia
    for col in feature_cols:
        if col not in df.columns:
            # Hitung feature turunan jika diperlukan
            if col == 'dept_promo_rate':
                df['dept_promo_rate'] = 0.08  # default value
            elif col == 'dept_avg_training':
                df['dept_avg_training'] = 65
            elif col == 'dept_avg_rating':
                df['dept_avg_rating'] = 3.5
            elif col == 'rating_x_training':
                df['rating_x_training'] = df['previous_year_rating'] * df['avg_training_score']
            elif col == 'total_score':
                df['total_score'] = df['avg_training_score'] * df['previous_year_rating'] + df['awards_won?'] * 50
            elif col == 'experience_rating':
                df['experience_rating'] = df['length_of_service'] * df['previous_year_rating']
            elif col == 'training_per_service':
                df['training_per_service'] = df['avg_training_score'] / (df['length_of_service'] + 1)
            elif col == 'high_performer':
                df['high_performer'] = ((df['previous_year_rating'] >= 4) & (df['avg_training_score'] >= 75)).astype(int)
            elif col == 'awards_x_rating':
                df['awards_x_rating'] = df['awards_won?'] * df['previous_year_rating']
            elif col == 'rating_squared':
                df['rating_squared'] = df['previous_year_rating'] ** 2
            elif col == 'training_high':
                df['training_high'] = (df['avg_training_score'] >= 80).astype(int)
            elif col == 'achievement_score':
                df['achievement_score'] = df['awards_won?'] + df['training_high'] + (df['previous_year_rating'] >= 4).astype(int)
            elif col == 'training_vs_dept':
                df['training_vs_dept'] = df['avg_training_score'] - df.get('dept_avg_training', 65)
            elif col == 'rating_vs_dept':
                df['rating_vs_dept'] = df['previous_year_rating'] - df.get('dept_avg_rating', 3.5)
            else:
                # Jika feature tidak dikenal, isi dengan 0
                df[col] = 0
    
    # Predict probability
    proba = model.predict_proba(df[feature_cols])[:, 1]
    df['promo_probability'] = proba
    
    # Build decision matrix
    dm = df[['promo_probability', 'length_of_service', 'awards_won?',
             'avg_training_score', 'previous_year_rating']].values.astype(float)
    scores = topsis(dm, weights, benefit)
    df['TOPSIS_Score'] = scores
    df['Rank'] = pd.Series(scores).rank(ascending=False).astype(int).values
    
    return df.sort_values('Rank')

# 5 kandidat uji dengan profil bervariasi
test_candidates = [
    {'name':'Andi (Muda Berprestasi)','department':'Sales & Marketing','region':'region_7',
     'education':"Master's & above",'gender':'m','recruitment_channel':'sourcing',
     'no_of_trainings':3,'age':28,'previous_year_rating':5.0,
     'length_of_service':3,'awards_won?':1,'avg_training_score':92},
    {'name':'Budi (Senior Rata-rata)','department':'Operations','region':'region_2',
     'education':"Bachelor's",'gender':'m','recruitment_channel':'other',
     'no_of_trainings':1,'age':45,'previous_year_rating':3.0,
     'length_of_service':18,'awards_won?':0,'avg_training_score':55},
    {'name':'Citra (Mid KPI Tinggi)','department':'Technology','region':'region_22',
     'education':"Master's & above",'gender':'f','recruitment_channel':'sourcing',
     'no_of_trainings':2,'age':35,'previous_year_rating':4.0,
     'length_of_service':8,'awards_won?':1,'avg_training_score':78},
    {'name':'Dewi (Junior Potensial)','department':'Analytics','region':'region_13',
     'education':"Bachelor's",'gender':'f','recruitment_channel':'sourcing',
     'no_of_trainings':4,'age':26,'previous_year_rating':4.0,
     'length_of_service':2,'awards_won?':0,'avg_training_score':85},
    {'name':'Eko (Senior Berpengalaman)','department':'HR','region':'region_4',
     'education':"Bachelor's",'gender':'m','recruitment_channel':'other',
     'no_of_trainings':1,'age':50,'previous_year_rating':3.0,
     'length_of_service':22,'awards_won?':1,'avg_training_score':60},
]

result = predict_and_rank(test_candidates, model, le_dict, feature_cols, weights, benefit)
print("=== Smart HR Promotion Dashboard ===")
print(result[['name','promo_probability','length_of_service','awards_won?',
              'avg_training_score','TOPSIS_Score','Rank']].to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 5))
res = result.sort_values('Rank')
colors = ['#2ecc71','#27ae60','#f39c12','#e67e22','#e74c3c']
bars = ax.barh(range(len(res)), res['TOPSIS_Score'], color=colors)
ax.set_yticks(range(len(res)))
ax.set_yticklabels(res['name'])
ax.set_xlabel('TOPSIS Priority Score')
ax.set_title('Final Promotion Ranking - Smart HR Dashboard')
ax.invert_yaxis()
for i, (v, r) in enumerate(zip(res['TOPSIS_Score'], res['Rank'])):
    ax.text(v+0.005, i, f'Rank #{r} | Prob: {res.iloc[i]["promo_probability"]:.1%}', va='center')
plt.tight_layout(); plt.savefig('final_ranking.png', dpi=150, bbox_inches='tight'); plt.show()
decision_matrix_sorted.to_csv('decision_matrix_result.csv')
result.to_csv('dashboard_simulation_result.csv', index=False)
print("Files exported:")
print("  - decision_matrix_result.csv")
print("  - dashboard_simulation_result.csv")
print("  - model_eval.png, shap_summary.png, shap_bar.png")
print("  - promotion_ranking.png, what_if_analysis.png, final_ranking.png")
print("\nNotebook selesai!")
