import pandas as pd
import numpy as np

def preprocess_training_csv(filepath):
    print("Processing Training.csv...")
    df = pd.read_csv(filepath)
    # The last column is 'prognosis', the rest are symptoms
    symptom_cols = df.columns[:-1]
    
    def get_symptoms(row):
        # Return a comma separated string of symptoms that have a 1
        symptoms = [col.replace('_', ' ') for col in symptom_cols if row[col] == 1]
        return ', '.join(symptoms)
        
    df['Symptoms_Text'] = df.apply(get_symptoms, axis=1)
    df = df.rename(columns={'prognosis': 'Disease'})
    
    # We only need Disease and Symptoms_Text
    return df[['Disease', 'Symptoms_Text']]

def preprocess_profile_csv(filepath):
    print("Processing Disease_symptom_and_patient_profile_dataset.csv...")
    df = pd.read_csv(filepath)
    
    # Filter positive cases only
    df = df[df['Outcome Variable'] == 'Positive']
    
    symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    
    def get_symptoms(row):
        symptoms = [col for col in symptom_cols if row[col] == 'Yes']
        return ', '.join(symptoms)
        
    df['Symptoms_Text'] = df.apply(get_symptoms, axis=1)
    
    # Keep patient profile data
    cols_to_keep = ['Disease', 'Symptoms_Text', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
    return df[cols_to_keep]

def preprocess_diseases_symptoms_csv(filepath):
    print("Processing Diseases_Symptoms.csv...")
    df = pd.read_csv(filepath)
    df = df.rename(columns={'Name': 'Disease', 'Symptoms': 'Symptoms_Text'})
    cols_to_keep = ['Disease', 'Symptoms_Text', 'Treatments']
    return df[cols_to_keep]

def preprocess_main_csv(filepath):
    print("Processing main.csv...")
    df = pd.read_csv(filepath)
    
    if 'frequency' in df.columns:
        df = df.drop(columns=['frequency'])
        
    symptom_cols = [col for col in df.columns if col != 'label']
    
    def clean_umls_string(s):
        if '_' in str(s):
            return s.split('_', 1)[1]
        return str(s)
        
    def get_symptoms(row):
        symptoms = [clean_umls_string(col) for col in symptom_cols if row[col] == 1.0 or row[col] == 1]
        return ', '.join(symptoms)
        
    df['Symptoms_Text'] = df.apply(get_symptoms, axis=1)
    
    def clean_label(label):
        first_disease = str(label).split('^')[0]
        return clean_umls_string(first_disease)
        
    df['Disease'] = df['label'].apply(clean_label)
    return df[['Disease', 'Symptoms_Text']]

def main():
    print("Loading and preprocessing datasets...")
    
    try:
        df1 = preprocess_training_csv('dataset/Training.csv')
    except Exception as e:
        print(f"Error processing Training.csv: {e}")
        df1 = pd.DataFrame()
        
    try:
        df2 = preprocess_profile_csv('dataset/Disease_symptom_and_patient_profile_dataset.csv')
    except Exception as e:
        print(f"Error processing Disease_symptom_and_patient_profile_dataset.csv: {e}")
        df2 = pd.DataFrame()
        
    try:
        df3 = preprocess_diseases_symptoms_csv('dataset/Diseases_Symptoms.csv')
    except Exception as e:
        print(f"Error processing Diseases_Symptoms.csv: {e}")
        df3 = pd.DataFrame()

    try:
        df4 = preprocess_main_csv('dataset/main.csv')
    except Exception as e:
        print(f"Error processing main.csv: {e}")
        df4 = pd.DataFrame()

    print("Merging datasets...")
    # Concatenate them all together
    merged_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    
    # Reorder columns logically
    cols = ['Disease', 'Symptoms_Text', 'Treatments', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
    # Ensure all columns exist even if some df failed
    for col in cols:
        if col not in merged_df.columns:
            merged_df[col] = np.nan
            
    merged_df = merged_df[cols]
    
    output_path = 'dataset/Merged_Master_Dataset.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"Success! Merged dataset saved to {output_path} with {len(merged_df)} rows.")

if __name__ == "__main__":
    main()
