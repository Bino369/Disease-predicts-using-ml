import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

def combine_text_features(row):
    parts = []
    if pd.notna(row['Symptoms_Text']) and str(row['Symptoms_Text']).strip() != "":
        parts.append(str(row['Symptoms_Text']))
    if pd.notna(row['Gender']) and str(row['Gender']).strip() != "":
        parts.append(str(row['Gender']))
    if pd.notna(row['Age']) and str(row['Age']).strip() != "":
        parts.append(f"age {row['Age']}")
    if pd.notna(row['Blood Pressure']) and str(row['Blood Pressure']).strip() != "":
        parts.append(f"{row['Blood Pressure']} blood pressure")
    if pd.notna(row['Cholesterol Level']) and str(row['Cholesterol Level']).strip() != "":
        parts.append(f"{row['Cholesterol Level']} cholesterol")
    return " ".join(parts)

def main():
    print("--- Training Unified NLP Model ---")
    df = pd.read_csv("dataset/Merged_Master_Dataset.csv")
    
    print(f"Loaded {len(df)} rows.")
    
    # Drop rows with missing Disease
    df = df.dropna(subset=['Disease'])
    
    print("Combining text features...")
    df['Combined_Text'] = df.apply(combine_text_features, axis=1)
    
    X = df['Combined_Text']
    y = df['Disease']
    
    print("Encoding labels and vectorizing text...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {acc:.4f}")
    
    print("Saving models to 'models/' directory...")
    joblib.dump(le, 'models/unified_label_encoder.pkl')
    joblib.dump(vectorizer, 'models/unified_vectorizer.pkl')
    joblib.dump(clf, 'models/unified_model.pkl')
    
    # Let's also save the treatments mapping so the GUI can look it up
    # We can create a dictionary mapping Disease -> Treatments (first non-NaN one we find)
    treatments_dict = {}
    for _, row in df.dropna(subset=['Treatments']).iterrows():
        disease = row['Disease']
        if disease not in treatments_dict:
            treatments_dict[disease] = row['Treatments']
            
    joblib.dump(treatments_dict, 'models/treatments_dict.pkl')
    
    print("Success! Unified model and dependencies saved.")

if __name__ == "__main__":
    main()
