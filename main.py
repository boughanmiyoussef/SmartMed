from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import pandas as pd
import pickle
import warnings
import ast
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ---------- Load Model & Data ---------- #
try:
    possible_model_paths = [
        'svc.pkl',
        'models/svc.pkl',
        '../svc.pkl'
    ]
    
    svc = None
    for path in possible_model_paths:
        if os.path.exists(path):
            svc = pickle.load(open(path, 'rb'))
            print(f"✅ Model loaded from: {path}")
            break
    
    if svc is None:
        print("❌ Could not find svc.pkl. Using dummy model.")
        from sklearn.svm import SVC
        svc = SVC()
        X_dummy = np.array([[0, 0], [1, 1]])
        y_dummy = np.array([0, 1])
        svc.fit(X_dummy, y_dummy)
    
    # Load datasets
    try:
        desc_df = pd.read_csv('datasets/description.csv')
        prec_df = pd.read_csv('datasets/precautions_df.csv')
        med_df = pd.read_csv('datasets/medications.csv')
        diet_df = pd.read_csv('datasets/diets.csv')
        work_df = pd.read_csv('datasets/workout_df.csv')
        
        # ========== DYNAMIC MAPPINGS GENERATION ==========
        print("🔧 Generating dynamic mappings from dataset...")
        
        # Load training dataset to get symptom names
        training_data = pd.read_csv('datasets/Training.csv')
        
        # 1. Generate symptoms_dict dynamically from column names (ONLY 132 original)
        symptom_columns = [col for col in training_data.columns if col != 'prognosis']
        symptoms_dict = {}
        for i, symptom in enumerate(symptom_columns):
            symptoms_dict[symptom] = i  # 'itching' -> 0
        
        # 2. Generate disease mapping using LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(training_data['prognosis'])
        idx2dis = {i: disease for i, disease in enumerate(le.classes_)}
        
        print(f"✅ Generated: {len(symptoms_dict)} symptom mappings (should be 132)")
        print(f"✅ Generated: {len(idx2dis)} disease mappings (should be 41)")
        print("✅ All datasets loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        # Create dummy dataframes for testing
        desc_df = pd.DataFrame({'Disease': ['Fungal infection', 'Allergy'], 
                                'Description': ['Dummy description 1', 'Dummy description 2']})
        prec_df = pd.DataFrame({'Disease': ['Fungal infection', 'Allergy'],
                                'Precaution_1': ['Precaution 1', 'Precaution 1'],
                                'Precaution_2': ['Precaution 2', 'Precaution 2']})
        med_df = pd.DataFrame({'Disease': ['Fungal infection', 'Allergy'],
                               'Medication': "['Medicine 1', 'Medicine 2']"})
        diet_df = pd.DataFrame({'Disease': ['Fungal infection', 'Allergy'],
                                'Diet': "['Diet 1', 'Diet 2']"})
        work_df = pd.DataFrame({'disease': ['Fungal infection', 'Allergy'],
                                'workout': "['Workout 1', 'Workout 2']"})
        
        # Create minimal dummy mappings
        symptoms_dict = {'itching': 0, 'skin_rash': 1}
        idx2dis = {0: 'Fungal infection', 1: 'Allergy'}
        
except Exception as e:
    print(f"❌ Error loading files: {e}")
    svc = None
    desc_df = prec_df = med_df = diet_df = work_df = None
    symptoms_dict = {}
    idx2dis = {}

# ---------- Symptom Formatting Function ---------- #
def format_symptom(symptom):
    """Convert symptom to standard format (underscore, lowercase)"""
    if not symptom or not isinstance(symptom, str):
        return None
    
    symptom = symptom.strip().lower()
    
    # Try direct match
    if symptom in symptoms_dict:
        return symptom
    
    # Try replacing spaces with underscores (for user input like "skin rash")
    symptom_underscore = symptom.replace(' ', '_')
    if symptom_underscore in symptoms_dict:
        return symptom_underscore
    
    # Try common variations
    variations = [
        symptom,
        symptom.replace(' ', '_'),
        symptom.replace('-', '_'),
        symptom.replace('(', '').replace(')', '')
    ]
    
    for var in variations:
        if var in symptoms_dict:
            return var
    
    return None

# ---------- String to List Parsing Function --------- #
def parse_string_to_list(value):
    """
    Parse a string representation of a list to an actual list.
    Handles cases like: "['item1', 'item2', 'item3']"
    """
    if pd.isna(value):
        return []
    
    if isinstance(value, list):
        return [str(item).strip() for item in value if pd.notna(item) and str(item).strip()]
    
    if isinstance(value, str):
        value = value.strip()
        
        if value.startswith('[') and value.endswith(']'):
            try:
                parsed_list = ast.literal_eval(value)
                if isinstance(parsed_list, list):
                    return [str(item).strip() for item in parsed_list if str(item).strip()]
            except (ValueError, SyntaxError):
                try:
                    value = value[1:-1]
                    items = [item.strip().strip("'\" ") for item in value.split(',') if item.strip()]
                    return items
                except:
                    return []
        
        return [value] if value else []
    
    return [str(value).strip()] if str(value).strip() else []

# ---------- Prediction Function ---------- #
def get_predicted_value(patient_symptoms):
    """Predict disease from symptoms list"""
    try:
        if svc is None:
            return "Model not available"
        
        input_vector = np.zeros(len(symptoms_dict))
        valid_symptoms_count = 0
        
        for item in patient_symptoms:
            formatted_symptom = format_symptom(item)
            
            if formatted_symptom and formatted_symptom in symptoms_dict:
                input_vector[symptoms_dict[formatted_symptom]] = 1
                valid_symptoms_count += 1
        
        if valid_symptoms_count == 0:
            return None
        
        predicted_idx = svc.predict([input_vector])[0]
        predicted_disease = idx2dis.get(predicted_idx, "Unknown Disease")
        
        return predicted_disease
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Prediction error"

# ---------- Helper Function ---------- #
def helper(dis):
    """Get all recommendations for a disease"""
    try:
        if desc_df is None:
            return "No description available.", [], [], [], []
        
        # Get description
        desc_rows = desc_df[desc_df['Disease'] == dis]['Description'].values
        desc = " ".join(desc_rows) if len(desc_rows) > 0 else "No description available."
        
        # Get precautions
        pre = []
        prec_row = prec_df[prec_df['Disease'] == dis]
        if not prec_row.empty:
            pre = [str(p).strip() for p in prec_row[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0] if pd.notna(p) and str(p).strip()]
        
        # Get medications
        med = []
        med_row = med_df[med_df['Disease'] == dis]['Medication']
        if not med_row.empty:
            med_value = med_row.values[0]
            med = parse_string_to_list(med_value)
        
        # Get diet recommendations
        die = []
        diet_row = diet_df[diet_df['Disease'] == dis]['Diet']
        if not diet_row.empty:
            diet_value = diet_row.values[0]
            die = parse_string_to_list(diet_value)
        
        # Get workout recommendations
        wrk = []
        work_row = work_df[work_df['disease'] == dis]['workout']
        if not work_row.empty:
            workout_value = work_row.values[0]
            wrk = parse_string_to_list(workout_value)
        
        return desc, pre, med, die, wrk
    except Exception as e:
        print(f"Error in helper function: {e}")
        return "No description available.", [], [], [], []

# ---------- Flask Routes ---------- #
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/doctor/<path:filename>')
def serve_doctor(filename):
    """Serve doctor image"""
    doctor_folder = os.path.join(app.root_path, 'doctor')
    return send_from_directory(doctor_folder, filename)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle symptom prediction with GET and POST"""
    if request.method == 'GET':
        return render_template('index.html')
    
    elif request.method == 'POST':
        try:
            # Get symptoms from the form
            symptoms_text = request.form.get('symptoms', '')
            
            if not symptoms_text:
                return render_template('index.html', 
                                     error='Please enter at least one symptom')
            
            # Split by comma and clean - FIXED: convert spaces to underscores
            user_symptoms = [s.strip() for s in symptoms_text.split(',')]
            user_symptoms = [symptom.strip("[]' \"").replace(' ', '_') for symptom in user_symptoms]
            
            # Get prediction
            predicted_disease = get_predicted_value(user_symptoms)
            
            if not predicted_disease:
                return render_template('index.html',
                                     error='No valid symptoms entered. Please check and try again.')
            
            if predicted_disease == "Unknown Disease":
                return render_template('index.html',
                                     error='Could not identify disease. Please try different symptoms.')
            
            # Get recommendations
            desc, pre, med, die, wrkout = helper(predicted_disease)
                        
            return render_template('index.html',
                                 disease=predicted_disease,
                                 description=desc,
                                 precautions=pre,
                                 medications=med,
                                 diets=die,
                                 workouts=wrkout,
                                 input_symptoms=symptoms_text)
        
        except Exception as e:
            return render_template('index.html',
                                 error=f'An error occurred: {str(e)}')

if __name__ == "__main__":
    # Run on default Flask port 5000 with debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)