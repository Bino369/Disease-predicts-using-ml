import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

def main():
    print("--- Training Patient Profile Models ---")
    df = pd.read_csv("dataset/Disease_symptom_and_patient_profile_dataset.csv")
    
    # Filter for positive cases only (predicting the disease they actually have)
    df = df[df['Outcome Variable'] == 'Positive']
    
    features = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
    target = 'Disease'
    
    X = df[features]
    y = df[target]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, 'models/profile_label_encoder.pkl')
    
    # Define categorical and numeric features
    cat_features = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level']
    num_features = ['Age']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # We will save the fitted preprocessor
    X_processed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, 'models/profile_preprocessor.pkl')
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)
    
    # Train models
    clf_dt = DecisionTreeClassifier(random_state=42)
    clf_dt.fit(X_train, y_train)
    joblib.dump(clf_dt, 'models/profile_decision_tree.pkl')
    
    clf_rf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf_rf.fit(X_train, y_train)
    joblib.dump(clf_rf, 'models/profile_random_forest.pkl')
    
    clf_svm = SVC(random_state=42, probability=True)
    clf_svm.fit(X_train, y_train)
    joblib.dump(clf_svm, 'models/profile_svm_model.pkl')
    
    print("Models trained and saved successfully in 'models/' directory.")

if __name__ == "__main__":
    main()
