from tkinter import *
from tkinter import ttk
import joblib
import os
import warnings
import pandas as pd

# Ignore scikit-learn warnings
warnings.filterwarnings('ignore')

# INITIALIZE & LOAD MODELS
print("--- Initializing Unified Prediction System ---")

def load_pkl_file(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found. Please run 'python train_unified_model.py' first.")
        return None
    return joblib.load(filepath)

# Load Unified Models
unified_model = load_pkl_file('models/unified_model.pkl')
vectorizer = load_pkl_file('models/unified_vectorizer.pkl')
le = load_pkl_file('models/unified_label_encoder.pkl')
treatments_dict = load_pkl_file('models/treatments_dict.pkl')

if all([unified_model, vectorizer, le, treatments_dict]):
    print("Unified models and transformers loaded successfully!")
else:
    print("WARNING: Some models failed to load. The application might not function correctly.")

# PREDICTION FUNCTION
def predict_unified():
    """
    Takes the text input, vectorizes it, and predicts the disease using the unified NLP model.
    """
    user_input = symptom_text.get("1.0", END).strip()
    
    if not user_input:
        result_text.config(text="Please enter symptoms or patient details")
        treatment_text.config(text="")
        return

    try:
        # Vectorize the input
        X_vectorized = vectorizer.transform([user_input])
        
        # Perform prediction
        prediction_encoded = unified_model.predict(X_vectorized)
        disease_name = le.inverse_transform(prediction_encoded)[0]
        
        # Update UI with results
        result_text.config(text=disease_name)
        
        # Look up treatment
        treatment = treatments_dict.get(disease_name, "No standard treatment recorded in database.")
        treatment_text.config(text=treatment)
        
    except Exception as e:
        result_text.config(text="Prediction Error")
        treatment_text.config(text=str(e))

# ---------------------------------------------------------------------------------------------------
# GUI SETUP
# ---------------------------------------------------------------------------------------------------
root = Tk()
root.title("All-in-One Disease Predictor")
root.configure(background='#f0f4f8')
root.geometry("850x700")

# Heading
header = Label(root, text="Unified Disease Prediction System", font=("Helvetica", 26, "bold"), fg="#2c3e50", bg="#f0f4f8")
header.pack(pady=30)

# Main Frame
main_frame = Frame(root, bg="#f0f4f8")
main_frame.pack(padx=50, fill=BOTH, expand=True)

# Name Section
name_frame = Frame(main_frame, bg="#f0f4f8")
name_frame.pack(fill=X, pady=10)

Label(name_frame, text="Patient Name:", font=("Helvetica", 14, "bold"), bg="#f0f4f8", fg="#34495e").pack(side=LEFT, padx=10)
patient_name = Entry(name_frame, font=("Helvetica", 14), width=30)
patient_name.pack(side=LEFT, padx=10)

# Input Section
Label(main_frame, text="Enter Symptoms, Age, Gender, and Profile Details:", font=("Helvetica", 14, "bold"), bg="#f0f4f8", fg="#34495e").pack(anchor=W, padx=10, pady=(20, 5))
Label(main_frame, text="(Example: 45 year old male, severe headache, high blood pressure, cough)", font=("Helvetica", 10, "italic"), bg="#f0f4f8", fg="#7f8c8d").pack(anchor=W, padx=10)

symptom_text = Text(main_frame, height=8, font=("Helvetica", 13), bd=2, relief=FLAT)
symptom_text.pack(fill=X, padx=10, pady=10)

# Predict Button
predict_btn = Button(root, text="Predict Disease & Treatment", command=predict_unified, font=("Helvetica", 16, "bold"), bg="#3498db", fg="black", padx=20, pady=10, cursor="hand2")
predict_btn.pack(pady=20)

# Results Section
result_frame = Frame(root, bg="white", bd=1, relief=SOLID)
result_frame.pack(padx=50, pady=20, fill=BOTH)

# Predicted Disease
disease_label_frame = Frame(result_frame, bg="white")
disease_label_frame.pack(fill=X, padx=20, pady=(20, 10))
Label(disease_label_frame, text="PREDICTED DISEASE:", font=("Helvetica", 12, "bold"), fg="#e74c3c", bg="white").pack(side=LEFT)
result_text = Label(disease_label_frame, text="---", font=("Helvetica", 16, "bold"), fg="#c0392b", bg="white")
result_text.pack(side=LEFT, padx=20)

# Treatment
treatment_label_frame = Frame(result_frame, bg="white")
treatment_label_frame.pack(fill=X, padx=20, pady=(0, 20))
Label(treatment_label_frame, text="RECOMMENDED TREATMENT:", font=("Helvetica", 12, "bold"), fg="#27ae60", bg="white").pack(anchor=NW)
treatment_text = Label(treatment_label_frame, text="---", font=("Helvetica", 11), fg="#2c3e50", bg="white", wraplength=700, justify=LEFT)
treatment_text.pack(anchor=NW, pady=5)

# Footer
Label(root, text="Algorithm: Random Forest Classifier with TF-IDF Vectorization", font=("Helvetica", 10, "bold"), bg="#f0f4f8", fg="#34495e").pack(side=BOTTOM, pady=(0, 5))
Label(root, text="Note: This is a machine learning prediction. Consult a professional for medical advice.", font=("Helvetica", 9), bg="#f0f4f8", fg="#95a5a6").pack(side=BOTTOM, pady=5)

root.mainloop()
