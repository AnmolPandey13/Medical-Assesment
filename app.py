# Medical_Centre/app.py
# (Includes fixes for 'None' sorting and missing History feature in prediction input)

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import csv
import numpy as np # Keep for potential type handling by pipeline

app = Flask(__name__)

# --- Globals ---
prediction_pipeline = None
df = None # Main dataset for initial lookup
learned_responses_cache = {} # In-memory cache based on simple lookup file
LEARNED_RESPONSES_CACHE_FILE = 'learned_responses.csv' # Keeps quick lookup file
LEARNED_RESPONSES_CACHE_FIELDNAMES = ['SymptomsKey', 'PredictedCondition', 'Advice', 'Tips']
LEARNED_DETAILED_FILE = 'learned_responses_detailed.csv' # File collecting detailed data
LEARNED_DETAILED_FIELDNAMES = [
    'SymptomsKey', 'PredictedCondition', 'Symptoms', 'Past Medical History', 'Age',
    'Gender', 'Has Hypertension', 'Has Diabetes', 'Has Asthma', 'Has Breathing Issues'
]

# --- Loading Functions ---
def load_pipeline():
    global prediction_pipeline
    try:
        pipeline_path = 'best_condition_prediction_pipeline.pkl'
        prediction_pipeline = joblib.load(pipeline_path)
        print(f"Prediction pipeline loaded successfully from {pipeline_path}.")
    except FileNotFoundError:
        print(f"Error: Prediction pipeline file '{pipeline_path}' not found. Ensure train_models.py has been run successfully after modifications.")
        prediction_pipeline = None
    except Exception as e:
        print(f"An unexpected error occurred during pipeline loading: {e}")
        prediction_pipeline = None

def load_dataset():
    global df
    try:
        # Ensure this path is correct
        df = pd.read_csv('vast_synthetic_patient_dataset_separate.csv')
        print("Main dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: Main dataset 'vast_synthetic_patient_dataset_separate.csv' not found.")
        df = None
    except Exception as e:
        print(f"An error occurred loading the main dataset for lookup: {e}")
        df = None

def load_learned_responses_cache(filepath):
    global learned_responses_cache
    learned_responses_cache = {}
    try:
        if not os.path.exists(filepath): # Create cache file if not exists
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=LEARNED_RESPONSES_CACHE_FIELDNAMES)
                writer.writeheader()
            print(f"'{filepath}' created.")
            return

        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or sorted(reader.fieldnames) != sorted(LEARNED_RESPONSES_CACHE_FIELDNAMES):
                 print(f"Warning: Header mismatch in '{filepath}'. Cache loading skipped.")
                 return
            for row in reader:
                symptoms_key = row.get('SymptomsKey')
                if symptoms_key:
                    learned_responses_cache[symptoms_key] = {
                        'PredictedCondition': row.get('PredictedCondition', 'N/A'),
                        'Advice': row.get('Advice', 'N/A'),
                        'Tips': row.get('Tips', 'N/A')
                    }
        print(f"Loaded {len(learned_responses_cache)} cache entries from '{filepath}'.")
    except Exception as e:
        print(f"Error loading cache file '{filepath}': {e}")
        learned_responses_cache = {}

def save_to_cache(filepath, cache_dict, new_entry):
    """Saves entry to simple cache file and updates dict."""
    cache_entry = {k: new_entry.get(k) for k in LEARNED_RESPONSES_CACHE_FIELDNAMES if k in new_entry}
    if not all(k in cache_entry for k in LEARNED_RESPONSES_CACHE_FIELDNAMES):
         print("Error: Cannot save to cache, missing required fields.")
         return
    symptoms_key = cache_entry['SymptomsKey']
    if not symptoms_key: return
    cache_dict[symptoms_key] = {k: v for k, v in cache_entry.items() if k != 'SymptomsKey'}
    try:
        file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=LEARNED_RESPONSES_CACHE_FIELDNAMES)
            if not file_exists: writer.writeheader()
            writer.writerow(cache_entry)
        print(f"Saved entry to cache file '{filepath}'.")
    except Exception as e:
        print(f"Error saving to cache file '{filepath}': {e}")
        if symptoms_key in cache_dict: del cache_dict[symptoms_key]

def save_detailed_learned_response(filepath, detailed_entry):
    """Appends a detailed learned response (full features + outcome) to the CSV for retraining."""
    if not all(key in detailed_entry for key in LEARNED_DETAILED_FIELDNAMES):
        print(f"Error: Missing keys in detailed_entry for saving. Required: {LEARNED_DETAILED_FIELDNAMES}")
        print(f"Got keys: {list(detailed_entry.keys())}")
        return
    symptoms_key = detailed_entry.get('SymptomsKey')
    if not symptoms_key:
         print("Error: Attempted to save detailed entry with empty SymptomsKey.")
         return
    try:
        file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=LEARNED_DETAILED_FIELDNAMES)
            if not file_exists: writer.writeheader()
            filtered_entry = {k: detailed_entry.get(k) for k in LEARNED_DETAILED_FIELDNAMES}
            writer.writerow(filtered_entry)
        print(f"Saved detailed entry for symptoms key '{symptoms_key}' to '{filepath}'.")
    except Exception as e:
        print(f"Error saving detailed entry to '{filepath}': {e}")

def normalize_symptoms(symptoms_str):
    """Normalizes symptoms string for consistent caching key."""
    if not isinstance(symptoms_str, str) or not symptoms_str.strip(): return None
    parts = []
    for part1 in symptoms_str.lower().split(','):
        for part2 in part1.split(';'):
            cleaned_part = part2.strip()
            if cleaned_part: parts.append(cleaned_part)
    if not parts: return None
    parts.sort()
    return ','.join(parts)

# --- Load data when the app starts ---
with app.app_context():
    load_pipeline()
    load_dataset()
    load_learned_responses_cache(LEARNED_RESPONSES_CACHE_FILE)

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main input form page (index.html)."""
    load_status = "OK"
    if prediction_pipeline is None:
        load_status = "Error: Prediction engine failed to load."
    elif df is None:
         load_status = "Warning: Main dataset for advice lookup failed to load."

    # --- Define options lists ---
    history_options_with_none = [
        'None', 'High Blood Pressure', 'Diabetes', 'Asthma', 'Breathing Issues', 'Allergies', 'Heart Problems',
        'Kidney Issues', 'Liver Issues', 'Cancer History', 'Smoking History',
        'Alcohol Use History', 'Infections', 'Stomach Digestive Issues', 'Joint Muscle Problems',
        'Mental Health Conditions', 'Thyroid Issues', 'Neurological Issues', 'Anemia',
        'Blood Clotting Issues', 'Skin Conditions', 'Recent Surgery', 'Recent Injury'
    ]
    symptom_options = [
        'Fever', 'Chills', 'Cough', 'Sore Throat', 'Runny Nose', 'Stuffy Nose', 'Headache', 'Dizziness', 'Fatigue',
        'Muscle Pain', 'Joint Pain', 'Swelling', 'Redness', 'Itching', 'Rash', 'Shortness of Breath',
        'Chest Pain', 'Palpitations', 'Leg Ankle Swelling', 'Abdominal Pain', 'Nausea', 'Vomiting',
        'Diarrhea', 'Constipation', 'Changes in Bowel Habits', 'Painful Urination', 'Frequent Urination',
        'Back Pain', 'Neck Pain', 'Stiff Neck', 'Vision Changes', 'Hearing Issues', 'Swallowing Difficulty',
        'Weight Loss', 'Weight Gain', 'Increased Thirst', 'Increased Hunger', 'Sleep Issues',
        'Memory Problems', 'Confusion', 'Tremors', 'Weakness', 'Numbness Tingling', 'Speech Problems',
        'Skin Changes Lesions Color', 'Hair Loss', 'Excessive Sweating', 'Mood Changes', 'Increased Anxiety'
    ]

    # --- Sort options, putting 'None' last for history --- ## CORRECTED SORTING LOGIC ##
    history_options_sorted = [opt for opt in history_options_with_none if opt != 'None']
    history_options_sorted.sort()
    history_options_final = history_options_sorted + ['None'] # Add 'None' last

    symptom_options.sort() # Sort symptoms normally

    # Pass the correctly ordered lists to the template
    return render_template('index.html',
                           load_status=load_status,
                           history_options=history_options_final, # Use the corrected list
                           symptom_options=symptom_options)


@app.route('/process', methods=['POST'])
def process_patient_data():
    """
    Handles form submission, checks cache, predicts if needed (using history text),
    looks up dataset info, saves to cache/detailed file if new, and renders results.
    """
    global prediction_pipeline, df, learned_responses_cache

    if request.method == 'POST':
        disclaimer_text = get_disclaimer()
        name = request.form.get('name', 'Patient').strip()

        try: # Form data retrieval and basic validation
            age_str = request.form.get('age', '').strip()
            gender = request.form.get('gender', 'Other').strip().capitalize()
            if gender not in ['Male', 'Female', 'Other']: gender = 'Other'
            selected_history = request.form.getlist('selected_history')
            other_history = request.form.get('other_history', '').strip()
            combined_history_list = [h for h in selected_history if h != 'None']
            if other_history:
                other_parts = [p.strip() for p in other_history.replace(';',',').split(',') if p.strip()]
                combined_history_list.extend(other_parts)
            final_medical_history = ', '.join(sorted(list(set(combined_history_list)))) if combined_history_list else "None provided"
            selected_symptoms = request.form.getlist('selected_symptoms')
            other_symptoms = request.form.get('other_symptoms', '').strip()
            combined_symptoms_list = selected_symptoms
            if other_symptoms:
                 other_parts = [p.strip() for p in other_symptoms.replace(';',',').split(',') if p.strip()]
                 combined_symptoms_list.extend(other_parts)
            final_symptoms_str = ', '.join(sorted(list(set(combined_symptoms_list))))
            hypertension_checked = 'hypertension' in request.form
            diabetes_checked = 'diabetes' in request.form
            if not final_symptoms_str: return render_template('results.html', name=name, error="Symptoms are required.", disclaimer=disclaimer_text)
            try:
                age = int(age_str);
                if not (0 < age <= 120): raise ValueError()
            except: return render_template('results.html', name=name, error="Invalid age entered.", disclaimer=disclaimer_text)
            print(f"\n--- New Request ---")
            print(f"Processing data for: {name}, {age}, {gender}")
            print(f"Symptoms: {final_symptoms_str}")
            print(f"History: {final_medical_history}")
            print(f"HTN: {hypertension_checked}, DM: {diabetes_checked}")
        except Exception as e:
            print(f"Error retrieving/combining form data: {e}")
            return render_template('results.html', name=name, error="Error processing form data.", disclaimer=disclaimer_text)

        # --- Check Cache (Quick Lookup) ---
        symptoms_key = normalize_symptoms(final_symptoms_str)
        print(f"Symptoms Normalized Key: {symptoms_key}")
        retrieved_condition, retrieved_advice, retrieved_tips = "Info unavailable.", "Info unavailable.", "Info unavailable."
        from_cache = False

        if symptoms_key and symptoms_key in learned_responses_cache:
            # --- Cache HIT ---
            print(f"Cache HIT for key: {symptoms_key}")
            cached_data = learned_responses_cache[symptoms_key]
            retrieved_condition = cached_data.get('PredictedCondition', 'N/A')
            retrieved_advice = cached_data.get('Advice', 'N/A')
            retrieved_tips = cached_data.get('Tips', 'N/A')
            from_cache = True
        elif symptoms_key:
            # --- Cache MISS ---
            print(f"Cache MISS for key: {symptoms_key}. Predicting...")
            from_cache = False

            # --- Prepare FULL features for Prediction (Includes History Text) ---
            history_lower = final_medical_history.lower() if final_medical_history != "None provided" else ""
            has_asthma = 1 if 'asthma' in history_lower else 0
            has_breathing_issues = 1 if ('breathing issues' in history_lower or 'breathing problems' in history_lower) else 0
            # Define the dictionary with ALL features the pipeline expects
            current_input_features = {
                'Symptoms': final_symptoms_str,
                # *** Ensure 'Past Medical History' key is present ***
                'Past Medical History': final_medical_history if final_medical_history != "None provided" else "",
                'Age': age,
                'Gender': gender,
                'Has Hypertension': 1 if hypertension_checked else 0,
                'Has Diabetes': 1 if diabetes_checked else 0,
                'Has Asthma': has_asthma,
                'Has Breathing Issues': has_breathing_issues
            }
            try:
                # Create DataFrame with keys matching training features
                input_df = pd.DataFrame([current_input_features])
                print("Input DataFrame prepared for prediction pipeline:\n", input_df.to_string())
            except Exception as e:
                 print(f"Error creating DataFrame for prediction: {e}")
                 return render_template('results.html', name=name, error="Internal error preparing data.", disclaimer=disclaimer_text)

            # --- Model Prediction & Dataset Lookup ---
            if prediction_pipeline:
                try:
                    print("Attempting prediction using the loaded pipeline...")
                    prediction_result = prediction_pipeline.predict(input_df) # Pipeline uses history text

                    if len(prediction_result) > 0:
                        predicted = prediction_result[0]; retrieved_condition = predicted
                        print(f"Pipeline prediction result: {predicted}")
                        if df is not None: # Lookup in main dataset
                            condition_rows = df.loc[df['Condition_1'] == predicted]
                            if not condition_rows.empty:
                                condition_row = condition_rows.iloc[0]
                                retrieved_advice = condition_row.get('Potential Advice', "Info unavailable.")
                                retrieved_tips = condition_row.get('General Tips', "Info unavailable.")
                                print("Found info in main dataset.")
                            else:
                                print(f"Predicted condition '{predicted}' not found in main dataset.")
                                retrieved_advice = "Advice not found."; retrieved_tips = "Tips not found."

                            # --- SAVE Detailed & Cache Entry ---
                            # Prepare data for detailed saving (includes all inputs)
                            detailed_entry = {**current_input_features, # Include all features used for prediction
                                              'SymptomsKey': symptoms_key,
                                              'PredictedCondition': predicted}
                            save_detailed_learned_response(LEARNED_DETAILED_FILE, detailed_entry)

                            # Prepare data for simple cache
                            cache_entry = {'SymptomsKey': symptoms_key,
                                           'PredictedCondition': predicted,
                                           'Advice': retrieved_advice,
                                           'Tips': retrieved_tips}
                            save_to_cache(LEARNED_RESPONSES_CACHE_FILE, learned_responses_cache, cache_entry)
                        else: print("Main lookup dataset not loaded.")
                    else: retrieved_condition = "Could not predict."; print("Prediction empty.")
                except Exception as e: print(f"Error during prediction/lookup: {e}"); retrieved_condition = "Prediction error."
            else: print("Prediction pipeline unavailable."); retrieved_condition = "Prediction engine not ready."
        else: print("Invalid symptoms key."); retrieved_condition = "Invalid symptoms."

        # --- Render Results ---
        return render_template('results.html',
                               name=name, age=age, gender=gender,
                               medical_history=final_medical_history,
                               hypertension= "Yes" if hypertension_checked else "No",
                               diabetes= "Yes" if diabetes_checked else "No",
                               symptoms=final_symptoms_str,
                               model_condition=retrieved_condition,
                               retrieved_advice=retrieved_advice,
                               retrieved_tips=retrieved_tips,
                               from_cache=from_cache,
                               disclaimer=disclaimer_text, error=None)
    return "Method Not Allowed", 405

def get_disclaimer():
    """Returns the standard disclaimer text."""
    return ("IMPORTANT DISCLAIMER: This tool provides preliminary information based on user input and data analysis. "
            "It is intended for informational purposes only and does NOT constitute medical advice, diagnosis, or treatment. "
            "Information from this tool should not replace consultation with a qualified healthcare professional. "
            "Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. "
            "Never disregard professional medical advice or delay in seeking it because of something you have read from this tool.")

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=5001, debug=True) # Using port 5001