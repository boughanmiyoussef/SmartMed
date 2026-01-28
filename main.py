from flask import Flask, render_template, request, jsonify
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
    # Load model
    model_paths = ['svc.pkl', 'models/svc.pkl']
    svc = None
    for path in model_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                svc = pickle.load(f)
            print(f"Model loaded from: {path}")
            break
    
    if svc is None:
        print("Warning: Using dummy model - svc.pkl not found")
        from sklearn.svm import SVC
        svc = SVC()
        X_dummy = np.array([[0, 0], [1, 1]])
        y_dummy = np.array([0, 1])
        svc.fit(X_dummy, y_dummy)
    
    # Load datasets
    datasets_path = 'datasets/'
    desc_df = pd.read_csv(os.path.join(datasets_path, 'description.csv'))
    prec_df = pd.read_csv(os.path.join(datasets_path, 'precautions_df.csv'))
    med_df = pd.read_csv(os.path.join(datasets_path, 'medications.csv'))
    diet_df = pd.read_csv(os.path.join(datasets_path, 'diets.csv'))
    work_df = pd.read_csv(os.path.join(datasets_path, 'workout_df.csv'))
    print("All datasets loaded successfully")
    
except Exception as e:
    print(f"Error loading files: {e}")
    svc = None
    desc_df = prec_df = med_df = diet_df = work_df = None

# ---------- Symptom Dictionary ---------- #
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

# Create mapping for space version symptoms
symptoms_space_to_underscore = {s.replace('_', ' '): s for s in symptoms_dict.keys()}

# ---------- Index to Disease Mapping ---------- #
idx2dis = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
    33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
    23: 'Hypertension', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
    36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
    25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal Positional Vertigo',
    2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# ---------- Helper Functions ---------- #
def format_symptom(symptom):
    """Convert symptom to standard format"""
    if not isinstance(symptom, str):
        return None
    
    symptom = symptom.strip().lower()
    
    if symptom in symptoms_dict:
        return symptom
    
    symptom_underscore = symptom.replace(' ', '_')
    if symptom_underscore in symptoms_dict:
        return symptom_underscore
    
    if symptom in symptoms_space_to_underscore:
        return symptoms_space_to_underscore[symptom]
    
    return None

def parse_string_to_list(value):
    """Parse string representation of list to actual list"""
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
            except:
                value = value[1:-1]
                items = [item.strip().strip("'\"") for item in value.split(',') if item.strip()]
                return items
        return [value] if value else []
    
    return [str(value).strip()] if str(value).strip() else []

def get_predicted_value(patient_symptoms):
    """Predict disease from symptoms"""
    try:
        if svc is None:
            return "Unknown"
        
        input_vector = np.zeros(len(symptoms_dict))
        
        for item in patient_symptoms:
            formatted_symptom = format_symptom(item)
            if formatted_symptom and formatted_symptom in symptoms_dict:
                input_vector[symptoms_dict[formatted_symptom]] = 1
        
        predicted_idx = svc.predict([input_vector])[0]
        return idx2dis.get(predicted_idx, "Unknown Disease")
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"

def get_disease_info(disease):
    """Get all information for a disease"""
    try:
        if desc_df is None:
            return "No description available.", [], [], [], []
        
        # Get description
        desc_rows = desc_df[desc_df['Disease'] == disease]['Description'].values
        desc = " ".join(desc_rows) if len(desc_rows) > 0 else "No description available."
        
        # Get precautions
        pre = []
        prec_row = prec_df[prec_df['Disease'] == disease]
        if not prec_row.empty:
            pre = [str(p).strip() for p in prec_row[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0] if pd.notna(p) and str(p).strip()]
        
        # Get medications
        med = []
        med_row = med_df[med_df['Disease'] == disease]['Medication']
        if not med_row.empty:
            med = parse_string_to_list(med_row.values[0])
        
        # Get diet
        die = []
        diet_row = diet_df[diet_df['Disease'] == disease]['Diet']
        if not diet_row.empty:
            die = parse_string_to_list(diet_row.values[0])
        
        # Get workout
        wrk = []
        work_row = work_df[work_df['disease'] == disease]['workout']
        if not work_row.empty:
            wrk = parse_string_to_list(work_row.values[0])
        
        return desc, pre, med, die, wrk
        
    except Exception as e:
        print(f"Error getting disease info: {e}")
        return "No information available.", [], [], [], []

# ---------- Flask Routes ---------- #
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symptoms_text = request.form.get('symptoms', '')
        
        if not symptoms_text:
            return render_template('index.html', error='Please enter symptoms')
        
        # Process symptoms
        user_symptoms = [s.strip() for s in symptoms_text.split(',')]
        user_symptoms = [s.strip("[]'\"") for s in user_symptoms]
        
        # Get prediction
        predicted_disease = get_predicted_value(user_symptoms)
        
        if not predicted_disease:
            return render_template('index.html', error='Could not make prediction')
        
        # Get disease information
        desc, pre, med, die, wrk = get_disease_info(predicted_disease)
        
        return render_template('index.html',
                             disease=predicted_disease,
                             description=desc,
                             precautions=pre,
                             medications=med,
                             diets=die,
                             workouts=wrk,
                             input_symptoms=symptoms_text)
        
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        predicted_disease = get_predicted_value(symptoms)
        desc, pre, med, die, wrk = get_disease_info(predicted_disease)
        
        return jsonify({
            'disease': predicted_disease,
            'description': desc,
            'precautions': pre,
            'medications': med,
            'diets': die,
            'workouts': wrk
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': svc is not None,
        'datasets_loaded': all(df is not None for df in [desc_df, prec_df, med_df, diet_df, work_df])
    })

# ---------- Run Application ---------- #
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)