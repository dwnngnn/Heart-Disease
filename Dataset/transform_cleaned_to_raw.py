import pandas as pd
import os

path = 'heart_disease_dataset.csv'
df = pd.read_csv(path)

mapping = {
    'sex': {1: 'Male', 0: 'Female'},
    'cp': {1: 'Typical angina', 2: 'Atypical angina', 3: 'Non-anginal pain', 4: 'Asymptomatic'},
    'fbs': {1: 'True', 0: 'False'},
    'exang': {1: 'Yes', 0: 'No'},
    'slope': {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'},
    'smoking': {1: 'Smoking', 0: 'No Smoking'},
    'diabetes': {1: 'Has diabetes', 0: 'Does not have diabetes'},
    'target': {1: 'Risk Present', 0: 'Normal / No Risk'}
}

for col, m in mapping.items():
    if col in df.columns:
        df[col] = df[col].map(m).fillna(df[col])
    else:
        raise SystemExit(f'Missing column: {col}')

out = 'heart_disease.csv'
df.to_csv(out, index=False)
print('created', os.path.abspath(out))
