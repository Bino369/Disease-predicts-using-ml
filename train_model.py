import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifierman
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

def evaluate_model(name, y_true, y_pred):
    """Utility function to print evaluation metrics cleanly."""
    print(f"\n--- {name} Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    print("============================================")
    print("  Disease Prediction ML Training Pipeline  ")
    print("============================================")
    
    # 1. Load Data
    print("\n[1/5] Loading Data...")
    df = pd.read_csv("dataset/Training.csv")
    print(f"Original dataset shape: {df.shape}")

    # 2. Preprocess Data
    print("\n[2/5] Preprocessing Data (Cleaning)...")
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Dropping...")
        df.dropna(inplace=True)
    else:
        print("No missing values found.")

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows. Dropping...")
        df.drop_duplicates(inplace=True)
    else:
        print("No duplicate rows found.")
        
    print(f"Cleaned dataset shape: {df.shape}")

    # 3. Setup Features & Target
    print("\n[3/5] Setup Features, Encode, and Split...")
    X = df[l1]
    y = df['prognosis']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save the encoder for the prediction GUI
    joblib.dump(le, 'models/label_encoder.pkl')
    print(f"Total unique diseases encoded: {len(le.classes_)}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')

    # 4. Train Models & Evaluate
    print("\n[4/5] Training Models...")

    # Decision Tree
    clf_dt = tree.DecisionTreeClassifier(random_state=42)
    clf_dt.fit(X_train_scaled, y_train)
    evaluate_model("Decision Tree", y_test, clf_dt.predict(X_test_scaled))

    # Random Forest
    clf_rf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf_rf.fit(X_train_scaled, y_train)
    evaluate_model("Random Forest", y_test, clf_rf.predict(X_test_scaled))

    # SVM
    clf_svm = SVC(random_state=42, probability=True)
    clf_svm.fit(X_train_scaled, y_train)
    evaluate_model("Support Vector Machine (SVM)", y_test, clf_svm.predict(X_test_scaled))

    # 5. Save Models
    print("\n[5/5] Saving Trained Models...")
    joblib.dump(clf_dt, 'models/decision_tree.pkl')
    joblib.dump(clf_rf, 'models/random_forest.pkl')
    joblib.dump(clf_svm, 'models/svm_model.pkl')
    
    print("\nTraining Pipeline Complete! All models saved to 'models/' directory.")

if __name__ == "__main__":
    main()
