from tkinter import *
import numpy as np
import joblib
import os
import warnings

# Ignore scikit-learn warnings for unlabelled divisions etc. if any
warnings.filterwarnings('ignore')

# SYMPTOM LIST (Matches training data order exactly)
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

# INITIALIZE & LOAD MODELS (Instantly ready for prediction)
print("--- Initializing Prediction System ---")

def load_pkl_file(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found. Please run 'python train_model.py' first.")
        return None
    return joblib.load(filepath)

# Load Transformers
scaler = load_pkl_file('models/scaler.pkl')
le = load_pkl_file('models/label_encoder.pkl')

# Load ML Models
clf_dt = load_pkl_file('models/decision_tree.pkl')
clf_rf = load_pkl_file('models/random_forest.pkl')
clf_svm = load_pkl_file('models/svm_model.pkl')

if all([scaler, le, clf_dt, clf_rf, clf_svm]):
    print("All models and transformers loaded successfully!")
else:
    print("WARNING: Some models failed to load. The application might crash.")


# PREDICTION HELPER
def predict_disease(model):
    """
    Takes a pre-trained model, transforms the GUI inputs, and predicts the disease.
    """
    # 1. Reset symptom vector properly to 0s to avoid bleeding from previous clicks
    l2 = [0] * len(l1)
    
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    
    # Track if user selected at least one real symptom
    valid_symptoms_selected = False

    # 2. Map selected symptoms to the feature vector (l2)
    for k in range(0, len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1
                valid_symptoms_selected = True

    # Error handling: If no valid symptom was picked
    if not valid_symptoms_selected:
        t2.delete("1.0", END)
        t2.insert(END, "Please select symptoms")
        return

    # 3. Transform input for the model
    inputtest = [l2]
    try:
        inputtest_scaled = scaler.transform(inputtest)
    except Exception as e:
        t2.delete("1.0", END)
        t2.insert(END, "Scaling Error")
        return

    # 4. Perform instant prediction
    try:
        predict = model.predict(inputtest_scaled)
        predicted_encoded = predict[0]
        
        # Decode the numerical prediction back to string
        disease_name = le.inverse_transform([predicted_encoded])[0]
        
        # Get prediction confidence if the model supports it
        confidence_str = "--"
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(inputtest_scaled)[0]
            class_idx = list(model.classes_).index(predicted_encoded)
            confidence = proba[class_idx] * 100
            confidence_str = f"{confidence:.1f}%"
            
        # Display the result
        t2.delete("1.0", END)
        t2.insert(END, disease_name)
        confidenceLb.config(text=f"Confidence: {confidence_str}")
    except Exception as e:
        t2.delete("1.0", END)
        t2.insert(END, "Prediction Error")
        confidenceLb.config(text="Confidence: --")

# Button Wrappers
def DecisionTree():
    predict_disease(clf_dt)

def randomforest():
    predict_disease(clf_rf)

def svm():
    predict_disease(clf_svm)


# ---------------------------------------------------------------------------------------------------
# GUI SETUP
# ---------------------------------------------------------------------------------------------------
root = Tk()
root.title("Disease Predictor")
root.configure(background='#f0f4f8')
root.geometry("800x550")

# Entry variables
Symptom1 = StringVar()
Symptom1.set("Select Symptom")
Symptom2 = StringVar()
Symptom2.set("Select Symptom")
Symptom3 = StringVar()
Symptom3.set("Select Symptom")
Symptom4 = StringVar()
Symptom4.set("Select Symptom")
Symptom5 = StringVar()
Symptom5.set("Select Symptom")
Name = StringVar()

# Heading
w2 = Label(root, justify=CENTER, text="Disease Predictor using Machine Learning", fg="#2c3e50", bg="#f0f4f8")
w2.config(font=("Helvetica", 24, "bold"))
w2.grid(row=1, column=0, columnspan=3, pady=30)

# Labels
label_font = ("Helvetica", 12)

NameLb = Label(root, text="Name of the Patient", fg="#34495e", bg="#f0f4f8", font=label_font)
NameLb.grid(row=6, column=0, pady=10, padx=50, sticky=W)

S1Lb = Label(root, text="Symptom 1", fg="#34495e", bg="#f0f4f8", font=label_font)
S1Lb.grid(row=7, column=0, pady=10, padx=50, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="#34495e", bg="#f0f4f8", font=label_font)
S2Lb.grid(row=8, column=0, pady=10, padx=50, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="#34495e", bg="#f0f4f8", font=label_font)
S3Lb.grid(row=9, column=0, pady=10, padx=50, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="#34495e", bg="#f0f4f8", font=label_font)
S4Lb.grid(row=10, column=0, pady=10, padx=50, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="#34495e", bg="#f0f4f8", font=label_font)
S5Lb.grid(row=11, column=0, pady=10, padx=50, sticky=W)

# Result Label
destreeLb = Label(root, text="Predicted Disease:", fg="#c0392b", bg="#f0f4f8", font=("Helvetica", 14, "bold"))
destreeLb.grid(row=15, column=0, pady=30, padx=50, sticky=W)

# Entries
OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name, font=label_font, width=25)
NameEn.grid(row=6, column=1)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.config(width=20, font=label_font)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.config(width=20, font=label_font)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.config(width=20, font=label_font)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.config(width=20, font=label_font)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.config(width=20, font=label_font)
S5En.grid(row=11, column=1)

# Predict Buttons
dt = Button(root, text="Decision Tree", command=DecisionTree, font=("Helvetica", 14, "bold"), width=15)
dt.grid(row=8, column=2, padx=40)

rnf = Button(root, text="Random Forest", command=randomforest, font=("Helvetica", 14, "bold"), width=15)
rnf.grid(row=9, column=2, padx=40)

svm_btn = Button(root, text="SVM Predict", command=svm, font=("Helvetica", 14, "bold"), width=15)
svm_btn.grid(row=10, column=2, padx=40)

# Result textfield
t2 = Text(root, height=1, width=25, bg="white", fg="#e74c3c", font=("Helvetica", 16, "bold"), bd=0)
t2.grid(row=15, column=1 , padx=10, sticky=W)

# Confidence Label
confidenceLb = Label(root, text="Confidence: --", fg="#2980b9", bg="#f0f4f8", font=("Helvetica", 14, "bold"))
confidenceLb.grid(row=15, column=2, padx=10, sticky=W)

# Run UI
root.mainloop()
