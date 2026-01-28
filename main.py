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
    # Try different possible locations for svc.pkl
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
        
except Exception as e:
    print(f"❌ Error loading files: {e}")
    svc = None
    desc_df = prec_df = med_df = diet_df = work_df = None

# ---------- Symptom Dictionary (132 symptoms) ---------- #
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
    'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11,
    'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21,
    'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26,
    'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
    'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
    'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
    'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64,
    'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73,
    'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77,
    'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
    'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
    'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
    'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128,
    'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# Create a mapping from "space version" to "underscore version"
symptoms_space_to_underscore = {}
for symptom_underscore in symptoms_dict.keys():
    symptom_space = symptom_underscore.replace('_', ' ')
    symptoms_space_to_underscore[symptom_space] = symptom_underscore

# ---------- Index to Disease Mapping ---------- #
idx2dis = {
    0: '(vertigo) Paroymsal Positional Vertigo',
    1: 'AIDS',
    2: 'Acne',
    3: 'Alcoholic hepatitis',
    4: 'Allergy',
    5: 'Arthritis',
    6: 'Bronchial Asthma',
    7: 'Cervical spondylosis',
    8: 'Chicken pox',
    9: 'Chronic cholestasis',
    10: 'Common Cold',
    11: 'Dengue',
    12: 'Diabetes',
    13: 'Dimorphic hemmorhoids(piles)',
    14: 'Drug Reaction',
    15: 'Fungal infection',
    16: 'GERD',
    17: 'Gastroenteritis',
    18: 'Heart attack',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    23: 'Hypertension',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    26: 'Hypothyroidism',
    27: 'Impetigo',
    28: 'Jaundice',
    29: 'Malaria',
    30: 'Migraine',
    31: 'Osteoarthristis',
    32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic ulcer diseae',
    34: 'Pneumonia',
    35: 'Psoriasis',
    36: 'Tuberculosis',
    37: 'Typhoid',
    38: 'Urinary tract infection',
    39: 'Varicose veins',
    40: 'hepatitis A'
}

# ---------- Symptom Formatting Function ---------- #
def format_symptom(symptom):
    """Convert symptom to standard format (underscore, lowercase)"""
    if not symptom or not isinstance(symptom, str):
        return None
    
    symptom = symptom.strip().lower()
    
    # Try direct match
    if symptom in symptoms_dict:
        return symptom
    
    # Try replacing spaces with underscores
    symptom_underscore = symptom.replace(' ', '_')
    if symptom_underscore in symptoms_dict:
        return symptom_underscore
    
    # Try with underscore to space mapping
    if symptom in symptoms_space_to_underscore:
        return symptoms_space_to_underscore[symptom]
    
    # Try common variations
    variations = [
        symptom,
        symptom.replace(' ', '_'),
        symptom.replace('-', '_'),
        symptom.replace('.', ''),
        symptom.replace('(', '').replace(')', '')
    ]
    
    for var in variations:
        if var in symptoms_dict:
            return var
    
    return None

# ---------- String to List Parsing Function ---------- #
def parse_string_to_list(value):
    """
    Parse a string representation of a list to an actual list.
    Handles cases like: "['item1', 'item2', 'item3']"
    """
    if pd.isna(value):
        return []
    
    # If already a list, return it
    if isinstance(value, list):
        return [str(item).strip() for item in value if pd.notna(item) and str(item).strip()]
    
    # If it's a string
    if isinstance(value, str):
        value = value.strip()
        
        # Check if it looks like a Python list string representation
        if value.startswith('[') and value.endswith(']'):
            try:
                # Use ast.literal_eval for safe evaluation
                parsed_list = ast.literal_eval(value)
                if isinstance(parsed_list, list):
                    return [str(item).strip() for item in parsed_list if str(item).strip()]
            except (ValueError, SyntaxError):
                # If ast.literal_eval fails, try manual parsing
                try:
                    # Remove brackets
                    value = value[1:-1].strip()
                    # Split by commas, handling quotes
                    items = []
                    current = ''
                    in_quotes = False
                    quote_char = None
                    
                    for char in value:
                        if char in ['"', "'"]:
                            if in_quotes and char == quote_char:
                                in_quotes = False
                            elif not in_quotes:
                                in_quotes = True
                                quote_char = char
                            current += char
                        elif char == ',' and not in_quotes:
                            # Remove quotes from the item
                            item = current.strip().strip("'\"")
                            if item:
                                items.append(item)
                            current = ''
                        else:
                            current += char
                    
                    # Add the last item
                    if current.strip():
                        item = current.strip().strip("'\"")
                        if item:
                            items.append(item)
                    
                    return items
                except:
                    # Last resort: simple split
                    value = value[1:-1]  # Remove brackets
                    items = [item.strip().strip("'\" ") for item in value.split(',') if item.strip()]
                    return items
        
        # If it's not a list representation, return as single item list
        return [value] if value else []
    
    # For other types, convert to string and return as single item list
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
        
        # Get medications - using the new parsing function
        med = []
        med_row = med_df[med_df['Disease'] == dis]['Medication']
        if not med_row.empty:
            med_value = med_row.values[0]
            med = parse_string_to_list(med_value)
        
        # Get diet recommendations - using the new parsing function
        die = []
        diet_row = diet_df[diet_df['Disease'] == dis]['Diet']
        if not diet_row.empty:
            diet_value = diet_row.values[0]
            die = parse_string_to_list(diet_value)
        
        # Get workout recommendations - using the new parsing function
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
        # GET request - just show the form
        return render_template('index.html')
    
    elif request.method == 'POST':
        # POST request - process symptoms
        try:
            # Get symptoms from the form
            symptoms_text = request.form.get('symptoms', '')
            
            if not symptoms_text:
                return render_template('index.html', 
                                     error='Please enter at least one symptom')
            
            # Split by comma and clean
            user_symptoms = [s.strip() for s in symptoms_text.split(',')]
            user_symptoms = [symptom.strip("[]' \"") for symptom in user_symptoms]
            
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
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)