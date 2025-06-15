import random
import pandas as pd

# Expanded lists with more general terms
genders = ['Male', 'Female', 'Other']
medical_history = [
    'High Blood Pressure', 'Diabetes', 'Asthma', 'Breathing Issues', 'Allergies', 'Heart Problems',
    'Kidney Issues', 'Liver Issues', 'Cancer History', 'Smoking History',
    'Alcohol Use History', 'Infections', 'Stomach Digestive Issues', 'Joint Muscle Problems',
    'Mental Health Conditions', 'Thyroid Issues', 'Neurological Issues', 'Anemia',
    'Blood Clotting Issues', 'Skin Conditions', 'Recent Surgery', 'Recent Injury', 'None'
]

symptoms_list = [
    'Fever', 'Chills', 'Cough', 'Sore Throat', 'Runny Nose', 'Stuffy Nose', 'Headache', 'Dizziness', 'Fatigue',
    'Muscle Pain', 'Joint Pain', 'Swelling', 'Redness', 'Itching', 'Rash', 'Shortness of Breath',
    'Chest Pain', 'Palpitations', 'Leg Ankle Swelling', 'Abdominal Pain', 'Nausea', 'Vomiting',
    'Diarrhea', 'Constipation', 'Changes in Bowel Habits', 'Painful Urination', 'Frequent Urination',
    'Back Pain', 'Neck Pain', 'Stiff Neck', 'Vision Changes', 'Hearing Issues', 'Swallowing Difficulty',
    'Weight Loss', 'Weight Gain', 'Increased Thirst', 'Increased Hunger', 'Sleep Issues',
    'Memory Problems', 'Confusion', 'Tremors', 'Weakness', 'Numbness Tingling', 'Speech Problems',
    'Skin Changes Lesions Color', 'Hair Loss', 'Excessive Sweating', 'Mood Changes', 'Increased Anxiety'
]

# More general mapping of symptoms to potential conditions (no slashes)
condition_symptom_map = {
    'Flu': ['Fever', 'Chills', 'Cough', 'Sore Throat', 'Headache', 'Fatigue', 'Muscle Pain'],
    'Common Cold': ['Runny Nose', 'Stuffy Nose', 'Sore Throat', 'Cough', 'Headache', 'Fatigue'],
    'Respiratory Infection': ['Fever', 'Cough', 'Shortness of Breath', 'Chest Pain'],
    'Headache Disorder': ['Headache', 'Dizziness', 'Vision Changes'],
    'Digestive Issue': ['Abdominal Pain', 'Nausea', 'Vomiting', 'Diarrhea', 'Constipation'],
    'Urinary Tract Issue': ['Painful Urination', 'Frequent Urination', 'Back Pain'],
    'Joint Problem': ['Joint Pain', 'Swelling', 'Stiff Neck'], # Changed 'Stiffness' to 'Stiff Neck' for clarity and no slash
    'Muscle Strain': ['Muscle Pain', 'Swelling', 'Weakness'],
    'Allergic Reaction': ['Rash', 'Itching', 'Swelling', 'Shortness of Breath'],
    'Anemia': ['Fatigue', 'Weakness', 'Dizziness'],
    'Anxiety': ['Increased Anxiety', 'Palpitations', 'Sleep Issues'],
    'Depression': ['Mood Changes', 'Fatigue', 'Sleep Issues', 'Weight Loss', 'Weight Gain'], # Separated weight change
    'High Blood Pressure': ['Headache', 'Dizziness', 'Chest Pain'],
    'Diabetes': ['Increased Thirst', 'Increased Hunger', 'Frequent Urination', 'Fatigue', 'Vision Changes'],
    'Neurological Issue': ['Headache', 'Dizziness', 'Numbness Tingling', 'Weakness', 'Speech Problems'], # No slash
    'Skin Condition': ['Rash', 'Itching', 'Skin Changes Lesions Color'], # No slash
    'Thyroid Problem': ['Fatigue', 'Weight Loss', 'Weight Gain', 'Mood Changes', 'Sleep Issues'], # Separated weight change
    'Heart Problem': ['Chest Pain', 'Palpitations', 'Shortness of Breath', 'Leg Ankle Swelling'], # No slash
    'Asthma': ['Shortness of Breath', 'Cough', 'Wheezing'], # Assuming 'Wheezing' is a symptom
    'Breathing Issues': ['Shortness of Breath', 'Difficulty Breathing'] # Assuming 'Difficulty Breathing' is a symptom (add to symptoms_list if needed)
}

# More general treatments and advice (no slashes in advice)
condition_treatment_map = {
    'Flu': ('Rest, fluids, over the counter pain fever reducers', 'Stay home, avoid contact with others, consult a doctor if symptoms worsen'),
    'Common Cold': ('Rest, fluids, over the counter cold remedies', 'Get plenty of rest, stay hydrated'),
    'Respiratory Infection': ('Rest, fluids, consult a doctor for diagnosis and treatment may include medication', 'Avoid smoking, maintain good hygiene'),
    'Headache Disorder': ('Over the counter pain relievers, consult a doctor if severe or frequent', 'Identify and avoid potential triggers, manage stress'),
    'Digestive Issue': ('Over the counter remedies, dietary changes, consult a doctor if severe or persistent', 'Maintain a balanced diet, stay hydrated'),
    'Urinary Tract Issue': ('Increase fluid intake, consult a doctor antibiotics may be needed', 'Practice good hygiene'),
    'Joint Problem': ('Over the counter pain relievers, rest, physical therapy, consult a doctor if severe', 'Gentle exercise, maintain a healthy weight'),
    'Muscle Strain': ('Rest, ice, compression, elevation RICE, over the counter pain relievers', 'Avoid strenuous activity, gentle stretching'),
    'Allergic Reaction': ('Antihistamines mild, seek immediate medical help for severe reactions', 'Avoid known allergens'),
    'Anemia': ('Iron supplements, dietary changes, consult a doctor for diagnosis and treatment', 'Eat iron rich foods'),
    'Anxiety': ('Stress management techniques, therapy, medication if prescribed', 'Practice relaxation techniques, regular exercise'),
    'Depression': ('Therapy, medication if prescribed, lifestyle changes', 'Seek professional help, maintain a healthy lifestyle'),
    'High Blood Pressure': ('Lifestyle changes diet exercise, medication if prescribed', 'Monitor blood pressure regularly, follow doctor\'s advice'),
    'Diabetes': ('Dietary management, exercise, medication oral or insulin, regular monitoring', 'Follow a diabetic diet, regular exercise, monitor blood sugar'),
    'Neurological Issue': ('Consult a doctor for diagnosis and treatment may include medication therapy', 'Follow doctor\'s recommendations'),
    'Skin Condition': ('Topical treatments, consult a doctor for diagnosis and treatment', 'Maintain good skin hygiene, avoid irritants'),
    'Thyroid Problem': ('Medication hormone replacement or antithyroid, regular monitoring', 'Follow doctor\'s recommendations'),
    'Heart Problem': ('Lifestyle changes, medication, medical procedures depending on the condition', 'Follow doctor\'s recommendations, maintain a healthy lifestyle'),
    'Asthma': ('Inhalers bronchodilators corticosteroids', 'Avoid triggers, use inhaler as prescribed'),
    'Breathing Issues': ('Consult a doctor for diagnosis and treatment', 'Avoid irritants, follow doctor\'s recommendations')
}

def get_relevant_conditions(symptoms):
    possible_conditions = set()
    for condition, condition_symptoms in condition_symptom_map.items():
        if any(sym in symptoms for sym in condition_symptoms):
            possible_conditions.add(condition)
    return list(possible_conditions)

def get_symptoms(num_symptoms=random.randint(1, 3)):
    return random.sample(symptoms_list, min(num_symptoms, len(symptoms_list)))

# Generate dataset
data = []
num_records = 10000
max_conditions = 3

for i in range(num_records):
    age = random.randint(1, 95)
    gender = random.choice(genders)
    has_hypertension = random.choice([0, 1])
    has_diabetes = random.choice([0, 1])
    has_asthma = 1 if 'Asthma' in random.sample(medical_history, random.randint(0, 2)) else 0
    has_breathing_issues = 1 if 'Breathing Issues' in random.sample(medical_history, random.randint(0, 2)) else 0
    history_list = [hist for hist in random.sample(medical_history, random.randint(0, 2)) if hist not in ['Asthma', 'Breathing Issues']]
    history = ', '.join(history_list)
    symptoms = get_symptoms()
    symptoms_str = ', '.join(symptoms)

    relevant_conditions = get_relevant_conditions(symptoms)
    if has_asthma and 'Asthma' not in relevant_conditions:
        relevant_conditions.append('Asthma')
    if has_breathing_issues and 'Breathing Issues' not in relevant_conditions:
        relevant_conditions.append('Breathing Issues')

    record = [i + 1, gender, age, has_hypertension, has_diabetes, has_asthma, has_breathing_issues, history, symptoms_str]

    # Add separate columns for conditions
    for j in range(max_conditions):
        if j < len(relevant_conditions):
            record.append(relevant_conditions[j])
        else:
            record.append(None)

    # Add separate columns for advice and tips for the first condition
    if relevant_conditions:
        advice, tips = condition_treatment_map.get(relevant_conditions[0], ('General supportive care', 'Consult a doctor'))
        record.append(advice)
        record.append(tips)
    else:
        record.append(None)
        record.append(None)

    data.append(record)

# Create column names
columns = ['ID', 'Gender', 'Age', 'Has Hypertension', 'Has Diabetes', 'Has Asthma', 'Has Breathing Issues', 'Past Medical History', 'Symptoms']
condition_cols = [f'Condition_{i+1}' for i in range(max_conditions)]
treatment_cols = ['Potential Advice', 'General Tips']
final_columns = columns + condition_cols + treatment_cols

df = pd.DataFrame(data, columns=final_columns)

df.to_csv('vast_synthetic_patient_dataset_separate.csv', index=False)
print(f"Generated a vast dataset with {num_records} records and separate columns (no slashes in conditions or symptoms).")