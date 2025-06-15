# train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# Note: MultinomialNB typically prefers non-negative features like raw counts or TF-IDF alone.
# Performance might vary on combined scaled/TF-IDF data. Keep it for comparison or remove if problematic.
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Use imblearn pipeline for SMOTE
import joblib
import numpy as np

def train_and_evaluate_models(df):
    print("Preparing features and target...")
    # --- Feature Selection ---
    # ADD 'Past Medical History' as a text feature
    feature_cols = [
        'Symptoms',             # Text 1
        'Past Medical History', # Text 2
        'Age',                  # Numerical
        'Gender',               # Categorical
        'Has Hypertension',     # Numerical (0/1)
        'Has Diabetes',         # Numerical (0/1)
        'Has Asthma',           # Numerical (0/1)
        'Has Breathing Issues'  # Numerical (0/1)
    ]
    target_col = 'Condition_1' # Still predicting the first condition

    # --- Handle Missing Text Data ---
    # Fill NaN in text columns with empty string BEFORE splitting
    df['Symptoms'] = df['Symptoms'].fillna('')
    df['Past Medical History'] = df['Past Medical History'].fillna('')

    X = df[feature_cols]
    y = df[target_col]

    # --- Handle Missing Target ---
    valid_indices = df[target_col].notna()
    X = X[valid_indices].copy() # Use .copy() to avoid SettingWithCopyWarning
    y = y[valid_indices].copy()
    print(f"Using {len(y)} records after removing missing target values.")

    if X.empty or y.empty:
        print("No data left after handling missing values. Exiting.")
        return

    # --- Split Data ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Data split successfully (stratified).")
    except ValueError:
        print("Could not stratify split. Splitting without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # --- Preprocessing Pipeline ---
    # Define column types
    numerical_features = ['Age', 'Has Hypertension', 'Has Diabetes', 'Has Asthma', 'Has Breathing Issues']
    categorical_features = ['Gender']
    symptom_text_features = 'Symptoms' # Pass column name string directly to ColumnTransformer
    history_text_features = 'Past Medical History' # Pass column name string directly

    # Create preprocessing pipelines for each data type
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Set sparse_output based on model needs
    ])

    # Define TF-IDF Vectorizers (potentially different settings)
    symptom_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1,2)) # Example settings
    history_vectorizer = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1,1)) # Example settings

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            # Apply TF-IDF to 'Symptoms' column
            ('sym_tfidf', symptom_vectorizer, symptom_text_features),
            # Apply TF-IDF to 'Past Medical History' column
            ('hist_tfidf', history_vectorizer, history_text_features)
        ],
        remainder='drop' # Drop any columns not specified
        # Consider setting sparse_threshold=0.0 if models struggle with mixed sparse/dense,
        # though it increases memory usage significantly. Default usually works.
    )
    print("ColumnTransformer preprocessing pipeline created (includes history TF-IDF).")

    # --- Models ---
    models = {
        'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42, max_iter=1000),
        'SVC': SVC(random_state=42, probability=True), # Probability=True useful but slower
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42, n_jobs=-1), # Use multiple cores
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1)
    }

    results = {}
    best_accuracy = 0
    best_model_name = None
    best_pipeline = None

    # --- Training and Evaluation Loop ---
    for name, model in models.items():
        print(f"Training and evaluating {name}...")

        # Create full pipeline: Preprocessing -> SMOTE -> Classifier
        full_pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            # Apply SMOTE only after features are numerical
            ('smote', SMOTE(random_state=42, k_neighbors=min(5, len(y_train.unique())-1) if len(y_train.unique()) > 1 else 1)), # Adjust k_neighbors for small classes
            ('classifier', model)
        ])

        try:
            full_pipeline.fit(X_train, y_train)
            y_pred = full_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            results[name] = {'accuracy': accuracy, 'classification_report': "Report omitted for brevity" if len(y.unique()) > 50 else report}
            print(f"{name} Accuracy: {accuracy:.4f}")
            if len(y.unique()) <= 50: # Print report only if classes are manageable
                 print(f"{name} Classification Report:\n{report}")
            print("-" * 50)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                best_pipeline = full_pipeline # Save the best *full* pipeline object
        except Exception as e:
            print(f"!!! Failed to train/evaluate {name}: {e}")
            # If SMOTE fails due to few samples in a class, might need to handle that class or adjust SMOTE params
            if "samples is less than k_neighbors" in str(e):
                 print("SMOTE failed possibly due to a class having too few samples.")
            print("-" * 50)

    print(f"\nBest Performing Model Pipeline: {best_model_name} with Accuracy: {best_accuracy:.4f}")

    # --- Save the Best Pipeline ---
    if best_pipeline:
        joblib.dump(best_pipeline, 'best_condition_prediction_pipeline.pkl')
        print("Best full prediction pipeline (preprocessing + model) saved as 'best_condition_prediction_pipeline.pkl'.")
    else:
        print("No model pipeline trained successfully.")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        df = pd.read_csv('vast_synthetic_patient_dataset_separate.csv')
        train_and_evaluate_models(df)
    except FileNotFoundError:
        print("Error: 'vast_synthetic_patient_dataset_separate.csv' not found. Please run generate_dataset.py first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")