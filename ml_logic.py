# ml_logic.py
import pandas as pd
import numpy as np
import traceback
import os
import uuid
import joblib # For saving models
from scipy.stats import skew, normaltest, spearmanr
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelEncoder, StandardScaler, OrdinalEncoder, RobustScaler, FunctionTransformer # Added FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline # Regular pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
)
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_selection import SelectFromModel
from numpy import logspace
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple

# Imbalanced-learn imports (Keep as is)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("ERROR: 'imbalanced-learn' library not found. Classification imbalance handling will be skipped. Please install it: pip install imbalanced-learn")
    ImbPipeline = Pipeline # Fallback to regular pipeline
    class SMOTE(BaseEstimator): # Dummy SMOTE that does nothing
        def __init__(self, random_state=None, k_neighbors=5): pass
        def fit_resample(self, X, y): return X, y
        def _fit_resample(self, X, y): return X, y
        def fit(self, X, y=None): return self
        def transform(self, X): return X
        def fit_transform(self, X, y=None): return self.fit(X,y).transform(X)


# ML Models (Added GaussianNB)
from sklearn.linear_model import (
    LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression,
    BayesianRidge
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB # <-- Added GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Feature Engine components (Keep as is)
try:
    from feature_engine.encoding import RareLabelEncoder
    from feature_engine.outliers import Winsorizer
except ImportError:
    print("Warning: Feature-engine components (RareLabelEncoder, Winsorizer) not found. Using basic fallbacks.")
    class RareLabelEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, tol=0.01, n_categories=5, replace_with="Rare", ignore_format=True): pass
        def fit(self, X, y=None): return self
        def transform(self, X): return X
    class Winsorizer(BaseEstimator, TransformerMixin):
         def __init__(self, capping_method=None, tail=None, fold=None): pass
         def fit(self, X, y=None): return self
         def transform(self, X): return X

# --- Model Save Directory --- (Keep as is)
MODEL_SAVE_DIR = "saved_models"

# --- Helper Functions (Keep as is) ---
def is_monotonic_trend(series):
    try:
        numeric_series = pd.to_numeric(series.dropna(), errors='coerce').dropna()
        if numeric_series.empty: return False
        return numeric_series.is_monotonic_increasing or numeric_series.is_monotonic_decreasing
    except Exception: return False

def is_datetime_or_year_col(series):
    if pd.api.types.is_datetime64_any_dtype(series): return True
    if series.dtype == 'object':
        try: pass # Keep original logic for object type date detection if needed
        except (ValueError, TypeError, IndexError): pass
    if np.issubdtype(series.dtype, np.number):
        non_null = series.dropna()
        if pd.api.types.is_float_dtype(non_null.dtype):
             if not non_null.apply(lambda x: x.is_integer() if pd.notna(x) else True).all(): return False
             non_null = non_null.loc[non_null.notna()].astype(int)
        if pd.api.types.is_integer_dtype(non_null.dtype):
             is_likely_year = non_null.between(1800, 2100)
             if non_null.empty: return False
             return is_likely_year.mean() > 0.8
    return False

# --- DateTransformer Class --- (Keep as is)
class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_original=True, extract_month=True, extract_dayofweek=True, errors='coerce'):
        self.drop_original=drop_original; self.extract_month=extract_month; self.extract_dayofweek=extract_dayofweek; self.errors=errors
    def fit(self, X, y=None):
        self.date_cols_ = X.columns.tolist(); self.feature_names_ = []
        for col in self.date_cols_:
            col_str = str(col)
            if self.extract_month: self.feature_names_.append(col_str + "_month")
            if self.extract_dayofweek: self.feature_names_.append(col_str + "_dayofweek")
            if not self.drop_original: self.feature_names_.append(col_str)
        return self
    def transform(self, X):
        X = X.copy(); X_out = pd.DataFrame(index=X.index)
        for col in self.date_cols_:
            dt_series = pd.to_datetime(X[col], errors=self.errors); col_str = str(col)
            if self.extract_month: X_out[col_str + "_month"] = dt_series.dt.month.fillna(-1).astype(int)
            if self.extract_dayofweek: X_out[col_str + "_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(int)
            if not self.drop_original: X_out[col_str] = dt_series
        return X_out
    def get_feature_names_out(self, input_features=None): return np.array(self.feature_names_)

# --- CORRECTED Feature Classifier Function --- (Keep as is from previous fix)
def enhanced_feature_classifier_v3(df, target_col=None, user_date_cols=None, rare_threshold=0.01):
    """
    Corrected version: Explicitly checks for boolean types first.
    Also checks if numeric coercion introduces NaNs, indicating non-numeric strings,
    and classifies such columns as categorical.
    """
    result = {
        "Numeric variables that require discretization": [],
        "Numeric variables that are normally distributed": [],
        "Numeric variables that have skew in them": [],
        "Nominal categorical variables": [],
        "Ordinal categorical variables": [],
        "Categorical variables with rare categories": [],
        "High-cardinality or ID-like categorical variables": [],
        "Datetime or Year-based variables": [],
        "Dropped quasi-constant columns": []
    }
    n_rows = df.shape[0]
    assigned = set()

    if user_date_cols:
        for col in user_date_cols:
            if col in df.columns and col != target_col:
                result["Datetime or Year-based variables"].append(col)
                assigned.add(col)

    for col in df.columns:
        if col == target_col or col in assigned:
            continue

        # Use the original series for initial NaN count and type checks
        original_series = df[col]
        original_nan_count = original_series.isna().sum()
        original_dtype = original_series.dtype

        col_data = original_series.dropna() # Use dropna for nunique, value_counts etc.

        if col_data.empty:
            if original_nan_count == n_rows:
                 print(f"Warning: Column '{col}' consists entirely of NaN values.")
                 result["Dropped quasi-constant columns"].append(col)
            else:
                 print(f"Warning: Column '{col}' became empty after dropping NaNs unexpectedly.")
                 result["Dropped quasi-constant columns"].append(col)
            assigned.add(col)
            continue

        n_unique = col_data.nunique()
        top_freq = col_data.value_counts(normalize=True, dropna=True)
        top_freq_ratio = top_freq.iloc[0] if not top_freq.empty else 1.0

        # Check for Boolean type FIRST
        is_bool_col = False
        if pd.api.types.is_bool_dtype(original_dtype): is_bool_col = True
        elif n_unique == 2 and set(col_data.unique()) == {0, 1}: is_bool_col = True
        elif n_unique == 2 and set(col_data.astype(str).str.lower().unique()) == {'true', 'false'}: is_bool_col = True

        if is_bool_col:
            print(f"Info: Column '{col}' classified as Boolean/Nominal Categorical.")
            result["Nominal categorical variables"].append(col)
            assigned.add(col)
            continue

        # Continue with other checks only if not boolean
        if is_datetime_or_year_col(original_series):
            result["Datetime or Year-based variables"].append(col)
            assigned.add(col)
            continue

        if original_series.nunique(dropna=False) <= 1 or top_freq_ratio > 0.98:
            result["Dropped quasi-constant columns"].append(col)
            assigned.add(col)
            continue

        if n_unique / n_rows > 0.9 or (hasattr(original_series, 'is_unique') and original_series.is_unique):
            result["High-cardinality or ID-like categorical variables"].append(col)
            assigned.add(col)
            continue

        # Check for non-numeric strings causing coercion NaNs
        numeric_coerced_series = pd.to_numeric(original_series, errors='coerce')
        coerced_nan_count = numeric_coerced_series.isna().sum()
        coercion_introduced_nans = coerced_nan_count > original_nan_count

        if coercion_introduced_nans:
            # Safely get an example of a failing value
            failing_indices = numeric_coerced_series.isna() & original_series.notna()
            example_value = original_series[failing_indices].iloc[0] if failing_indices.any() else "N/A"
            print(f"Info: Column '{col}' contains non-numeric strings (e.g., '{example_value}'). Classified as Nominal Categorical.")
            result["Nominal categorical variables"].append(col)
            assigned.add(col)
            continue

        # Attempt numeric classification ONLY if coercion didn't introduce NaNs
        is_numeric = False
        is_int_like = False
        numeric_col_data_post_coerce = numeric_coerced_series.dropna()

        if not numeric_col_data_post_coerce.empty and pd.api.types.is_numeric_dtype(numeric_col_data_post_coerce.dtype) and not pd.api.types.is_bool_dtype(numeric_col_data_post_coerce.dtype):
             valid_numeric_fraction_original = numeric_coerced_series.notna().sum() / (n_rows - original_nan_count) if (n_rows - original_nan_count) > 0 else 0
             if valid_numeric_fraction_original > 0.5:
                 is_numeric = True
                 valid_numeric = numeric_col_data_post_coerce
                 if pd.api.types.is_integer_dtype(valid_numeric.dtype): is_int_like = True
                 elif pd.api.types.is_float_dtype(valid_numeric.dtype): is_int_like = (valid_numeric == valid_numeric.round()).all()

        if is_numeric:
            current_n_unique = valid_numeric.nunique()
            if is_int_like and current_n_unique <= 20: result["Numeric variables that require discretization"].append(col)
            elif 3 <= current_n_unique <= 15: result["Numeric variables that require discretization"].append(col)
            else:
                try:
                    test_data = valid_numeric
                    if len(test_data) >= 20:
                        if pd.api.types.is_numeric_dtype(test_data) and not pd.api.types.is_bool_dtype(test_data):
                            stat, p_norm = normaltest(test_data)
                            skewness = skew(test_data)
                            if p_norm > 0.05 and abs(skewness) < 0.75: result["Numeric variables that are normally distributed"].append(col)
                            else: result["Numeric variables that have skew in them"].append(col)
                        else:
                             print(f"Warning: Column '{col}' passed numeric checks but dtype is {test_data.dtype} before normaltest. Classifying as skewed.")
                             result["Numeric variables that have skew in them"].append(col)
                    elif len(test_data) > 0:
                         result["Numeric variables that have skew in them"].append(col)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Error during normality/skew test for column '{col}': {e}. Classifying as skewed numeric.")
                    result["Numeric variables that have skew in them"].append(col)
            assigned.add(col)
            continue

        # If not classified yet, treat as categorical
        if col not in assigned:
             vc = original_series.astype(str).value_counts()
             is_rare = (vc < rare_threshold * n_rows) | (vc < 5)
             if is_rare.any(): result["Categorical variables with rare categories"].append(col)
             else: result["Nominal categorical variables"].append(col)
             assigned.add(col)
             continue

    final_result = {k: v for k, v in result.items() if v}
    return final_result

# --- Add a simple transformer to convert to string --- (Keep as is)
def _convert_to_string(X):
    """Helper function to convert DataFrame columns to string type"""
    if isinstance(X, pd.Series):
        X = X.to_frame()
    return X.astype(str)

ToStringTransformer = FunctionTransformer(_convert_to_string, validate=False)

# --- Preprocessing Pipeline Definitions --- (Keep as is from previous fix)
skewed_num_pipe = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median")),
    ("winsor", Winsorizer(capping_method='gaussian', tail='both', fold=1.5)),
    ("scale", RobustScaler())
])
norm_num_pipe = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="mean", add_indicator=True)),
    ("winsor", Winsorizer(capping_method='gaussian', tail='both', fold=3)),
    ("scale", StandardScaler())
])
disc_pipe = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median", add_indicator=True)),
    ("winsor", Winsorizer(capping_method='gaussian', tail='both', fold=1.5)),
    ("disc", KBinsDiscretizer(strategy="quantile", encode="ordinal", n_bins=5, subsample=200_000, random_state=42))
])
nom_cat_pipe = Pipeline(steps=[
    ("to_str", ToStringTransformer),
    ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
    ("rare", RareLabelEncoder(tol=0.01, n_categories=5, replace_with="Rare", ignore_format=True)),
    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown='ignore', min_frequency=10))
])
ord_cat_pipe = Pipeline(steps=[
    ("to_str", ToStringTransformer),
    ("imp", SimpleImputer(strategy="most_frequent", add_indicator=True)),
    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
rare_cat_pipe = Pipeline(steps=[
    ("to_str", ToStringTransformer),
    ("imp", SimpleImputer(strategy="constant", fill_value="Rare")),
    ("rare", RareLabelEncoder(tol=0.05, n_categories=2, replace_with="Rare", ignore_format=True)),
    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown='ignore', min_frequency=5))
])
date_pipe = Pipeline(steps=[
    ('dt', DateTransformer(extract_month=True, extract_dayofweek=True, drop_original=True))
])


# --- Function to Create the Full Preprocessor --- (Keep as is)
def create_preprocessor(df: pd.DataFrame, target_col: str, user_date_cols: Optional[List[str]] = None) -> Tuple[ColumnTransformer, List[List[str]]]:
    print("Classifying features...")
    X = df.drop(columns=[target_col], errors='ignore')
    if target_col in X.columns: X = X.drop(columns=[target_col])

    classified = enhanced_feature_classifier_v3(X, target_col=None, user_date_cols=user_date_cols)
    print(f"Feature classification results: {classified}")
    nom_cat_vars=classified.get("Nominal categorical variables",[]);ord_cat_vars=classified.get("Ordinal categorical variables",[]);rare_cat_vars=classified.get("Categorical variables with rare categories",[]);disc_num_vars=classified.get("Numeric variables that require discretization",[]);norm_num_vars=classified.get("Numeric variables that are normally distributed",[]);skewed_num_vars=classified.get("Numeric variables that have skew in them",[]);date_vars=classified.get("Datetime or Year-based variables",[])
    id_cols=classified.get("High-cardinality or ID-like categorical variables",[]);const_cols=classified.get("Dropped quasi-constant columns",[]);cols_to_drop=[id_cols, const_cols]
    transformers = []
    if nom_cat_vars: transformers.append(("nom", nom_cat_pipe, nom_cat_vars))
    if ord_cat_vars: transformers.append(("ord", ord_cat_pipe, ord_cat_vars))
    if rare_cat_vars: transformers.append(("rare", rare_cat_pipe, rare_cat_vars))
    if norm_num_vars: transformers.append(("norm", norm_num_pipe, norm_num_vars))
    if skewed_num_vars: transformers.append(("skew", skewed_num_pipe, skewed_num_vars))
    if disc_num_vars: transformers.append(("disc", disc_pipe, disc_num_vars))
    if date_vars: transformers.append(("dt", date_pipe, date_vars))

    if not transformers:
        print("Warning: No features classified for transformation.")
        preprocessor = ColumnTransformer(transformers=[], remainder='drop')
    else:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False
        )
    preprocessor.set_output(transform="pandas"); print("Preprocessor object created (unfitted).")
    return preprocessor, cols_to_drop


# --- Final Regression Training Function --- (Keep as is)
def auto_train_clean_data(X, y, preprocessor, search_type='random', n_iter=20, scoring_metric='r2'):
    """
    Final version: Handles leakage, includes feature selection, saves best model.
    Performance depends on data/features. Overfitting check (CV vs Test) needed by user.
    Replaced LinearRegression with BayesianRidge.
    """
    print("Splitting data into train/test sets...")
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set shape: {train_X.shape}, Test set shape: {test_X.shape}")
    print("Fitting preprocessor on TRAINING data only...")
    try:
        print("Data types going into preprocessor.fit:")
        print(train_X.dtypes.value_counts())
        preprocessor.fit(train_X, train_y)
        print("Preprocessor fitted successfully.")
        train_X_trans = preprocessor.transform(train_X)
        test_X_trans = preprocessor.transform(test_X)
        train_X_trans.columns = train_X_trans.columns.astype(str)
        test_X_trans.columns = test_X_trans.columns.astype(str)
        print(f"Transformed training data shape: {train_X_trans.shape}")
        if train_X_trans.empty and not X.empty:
             print("Warning: Preprocessing resulted in empty training data.")
    except Exception as e:
        print(f"Error during preprocessing: {e}"); traceback.print_exc()
        # Debugging prints kept
        if isinstance(e, TypeError) and 'argument must be uniformly strings or numbers' in str(e):
             print("\n--- Debugging Mixed Type Error ---") # ... (debug prints)
        elif isinstance(e, ValueError) and ('Cannot use' in str(e) or 'fill_value' in str(e)):
             print("\n--- Debugging Imputation Error ---")
             try:
                 print("  Input column types to preprocessor:")
                 print(train_X.info())
             except Exception as debug_e: print(f"  Error during debug imputation check: {debug_e}")
             print("--- End Debugging ---")
        raise ValueError(f"Preprocessing failed: {e}") from e

    # Feature Selection
    print("\nPerforming feature selection using RandomForestRegressor...")
    train_X_selected, test_X_selected = train_X_trans.copy(), test_X_trans.copy()
    if not train_X_trans.empty:
        try:
            selection_estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
            selector = SelectFromModel(estimator=selection_estimator, threshold='median')
            selector.fit(train_X_trans, train_y)
            selected_mask = selector.get_support()
            selected_features = train_X_trans.columns[selected_mask]

            if 0 < len(selected_features) < train_X_trans.shape[1]:
                 train_X_selected = train_X_trans[selected_features]
                 if not test_X_trans.empty: test_X_selected = test_X_trans[selected_features]
                 else: test_X_selected = pd.DataFrame(columns=selected_features)
                 print(f"Original feature count: {train_X_trans.shape[1]}")
                 print(f"Selected feature count: {train_X_selected.shape[1]}")
                 print(f"Selected features (first 20): {selected_features.tolist()[:20]}...")
            elif len(selected_features) == train_X_trans.shape[1]: print("Feature selection did not remove any features. Using all.")
            else:
                 print("Warning: Feature selection removed all features. Using all original transformed features.")
                 train_X_selected, test_X_selected = train_X_trans.copy(), test_X_trans.copy()
        except Exception as e:
            print(f"Error during feature selection: {e}. Using all features."); traceback.print_exc()
            train_X_selected, test_X_selected = train_X_trans.copy(), test_X_trans.copy()
    else:
        print("Skipping feature selection as transformed training data is empty.")


    # Model Training Loop
    models = {
        "BayesianRidge": (BayesianRidge(), {
             'alpha_1': logspace(-6, -1, 4), 'alpha_2': logspace(-6, -1, 4),
             'lambda_1': logspace(-6, -1, 4), 'lambda_2': logspace(-6, -1, 4)
             }, False),
        "Lasso": (Lasso(max_iter=5000, random_state=42), {'alpha': logspace(-4, 1, 6)}, False),
        "Ridge": (Ridge(random_state=42), {'alpha': logspace(-2, 3, 6)}, False),
        "ElasticNet": (ElasticNet(max_iter=5000, random_state=42), {'alpha': logspace(-4, 1, 5), 'l1_ratio': [0.1, 0.5, 0.9]}, False),
        "RandomForest": (RandomForestRegressor(random_state=42, n_jobs=-1), {'n_estimators': [100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5]}, False),
        "GradientBoosting": (GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}, False),
        "HistGradientBoosting": (HistGradientBoostingRegressor(random_state=42), {'learning_rate': [0.05, 0.1], 'max_iter': [100, 200]}, False),
        "LightGBM": (LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1), {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63]}, False),
        "MLPRegressor": (MLPRegressor(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10), {'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)], 'activation': ['relu', 'tanh'], 'solver': ['adam'], 'alpha': logspace(-5, 0, 5), 'learning_rate': ['constant', 'adaptive']}, True),
        "SVR": (SVR(), {'C': logspace(-1, 2, 4), 'gamma': ['scale'], 'kernel': ['rbf']}, True),
        "KNN": (KNeighborsRegressor(n_jobs=-1), {'n_neighbors': [3, 5, 9], 'weights': ['uniform', 'distance']}, True)
    }
    best_model_name = None; best_model_pipeline_obj = None; best_score = -np.inf; model_results = []

    if train_X_selected.empty:
        print("Error: Training data is empty after preprocessing and feature selection. Cannot train models.")
        best_model_details = {"name": "None", "error": "Training data became empty."}
        return best_model_details, pd.DataFrame(), None

    for name, (model, param_grid, needs_explicit_scaling) in models.items():
        print(f"\n🔍 Training model: {name}"); steps = [];
        if needs_explicit_scaling: steps.append(('explicit_scaler', StandardScaler()))
        steps.append(('model', model)); pipe = Pipeline(steps)
        pipeline_param_grid = {f"model__{k}": v for k, v in param_grid.items()}
        searcher = None

        if pipeline_param_grid:
            param_combinations=1;
            for key in pipeline_param_grid:
                 try: pipe.get_params()[key] ; param_combinations *= len(pipeline_param_grid[key])
                 except KeyError: print(f" Warning: Param '{key}' not found in pipeline for {name}. Skipping."); continue
            actual_n_iter = min(n_iter, param_combinations) if param_combinations > 0 else 0
            if actual_n_iter > 0:
                print(f"  (Using n_iter={actual_n_iter} for RandomizedSearch)")
                searcher = RandomizedSearchCV(pipe, pipeline_param_grid, n_iter=actual_n_iter, scoring=scoring_metric, cv=3, n_jobs=-1, random_state=42, error_score='raise')
            else: print(f"  Skipping RandomizedSearch for {name} due to zero valid parameter combinations or empty grid.")

        try:
            if searcher:
                searcher.fit(train_X_selected, train_y)
                best_estimator=searcher.best_estimator_; best_params=searcher.best_params_; cv_score=searcher.best_score_
            else:
                 pipe.fit(train_X_selected, train_y)
                 best_estimator=pipe; best_params={}; cv_score=None
        except Exception as e: print(f"  Failed to train {name}: {e}"); traceback.print_exc(); continue

        if test_X_selected.empty:
             print(f"  Warning: Test data is empty for model {name}. Skipping evaluation.")
             test_r2, test_mse = np.nan, np.nan
        else:
             pred_test = best_estimator.predict(test_X_selected)
             test_r2 = r2_score(test_y, pred_test)
             test_mse = mean_squared_error(test_y, pred_test)

        cleaned_params = {k.replace('model__', ''): v for k, v in best_params.items()}
        result_entry = {'Model': name, f'CV {scoring_metric}': cv_score if cv_score is not None else 'N/A', 'Test R2': test_r2, 'Test MSE': test_mse, 'Best Params': cleaned_params}
        model_results.append(result_entry); print(f"  📉 Test MSE: {test_mse:.4f}"); print(f"  📈 Test R²: {test_r2:.4f}")

        current_score_source = "CV"; current_score = cv_score if cv_score is not None and not np.isnan(cv_score) else -np.inf
        if (current_score == -np.inf or pd.isna(current_score)) and not pd.isna(test_r2):
             current_score = test_r2; current_score_source = "Test R2"
        elif pd.isna(current_score): current_score = -np.inf

        print(f"  Score used for comparison ({current_score_source}): {current_score:.4f}")
        if current_score > best_score:
            print(f"  New best model found: {name} (Score: {current_score:.4f} > Previous best: {best_score:.4f})")
            best_score = current_score; best_model_pipeline_obj = best_estimator; best_model_name = name

    # Save Best Model
    saved_filename = None; best_model_details = {}
    if best_model_pipeline_obj is not None:
        try:
            if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
            filename_base = f"model_regr_{best_model_name.replace(' ', '')}_{uuid.uuid4().hex[:8]}"
            saved_filename = f"{filename_base}.pkl"
            save_path = os.path.join(MODEL_SAVE_DIR, saved_filename)
            joblib.dump(best_model_pipeline_obj, save_path)
            print(f"\n✅ Best model '{best_model_name}' saved to: {save_path}")
            best_model_results_entry = next((item for item in model_results if item['Model'] == best_model_name), None)
            best_model_details = {"name": best_model_name, "pipeline_steps": str(best_model_pipeline_obj.steps), "best_params": best_model_results_entry['Best Params'] if best_model_results_entry else {}}
        except Exception as e:
            print(f"Error saving model {best_model_name}: {e}"); traceback.print_exc(); saved_filename = None
            best_model_details = {"name": best_model_name, "error": "Failed to save model."}
    else:
        print("\n⚠️ No best model found or trained successfully.")
        best_model_details = {"name": "None", "error": "No suitable model was found or trained."}

    if not model_results: print("Warning: No models trained."); return best_model_details, pd.DataFrame(), saved_filename
    results_df = pd.DataFrame(model_results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Test R2', ascending=False).reset_index(drop=True)
    print(f"\nRegression complete. Best Model: {best_model_name}")
    return best_model_details, results_df, saved_filename


# --- Final Classification Training Function --- (MODIFIED - Added GaussianNB)
def auto_train_classification(X, y, preprocessor, search_type='random', n_iter=20, scoring_metric='auto'):
    """
    Final version: Handles leakage, includes SMOTE, feature selection, saves best model.
    Performance depends on data/features. Overfitting/Imbalance check needed by user.
    Added GaussianNB classifier.
    """
    task_type = type_of_target(y); unique_classes = np.unique(y); n_classes = len(unique_classes); is_multiclass = n_classes > 2
    if scoring_metric == 'auto': scoring_metric = 'f1_weighted' if is_multiclass else 'f1'
    print(f"🔍 Detected task type: {task_type} ({'Multiclass' if is_multiclass else 'Binary'})")
    print(f"⚖️ Class distribution (Original Full Target): {dict(Counter(y))}"); print(f"🎯 Using scoring metric: {scoring_metric}")
    print("Splitting data into train/test sets (stratified)...")
    train_X, test_X, train_y, test_y = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    print(f"Train shapes: X={train_X.shape}, y={train_y.shape} | Test shapes: X={test_X.shape}, y={test_y.shape}")
    train_counts = dict(Counter(train_y)); print(f"Train target distribution: {train_counts}")
    print(f"Test target distribution: {dict(Counter(test_y))}")
    min_samples = min(train_counts.values()) if train_counts else 0
    smote_k_neighbors = min(3, max(1, min_samples - 1)) if min_samples > 1 else 1
    print(f"Minority class samples in train: {min_samples}. Using k_neighbors={smote_k_neighbors} for SMOTE.")

    # Fit Preprocessor
    print("Fitting preprocessor on TRAINING data only...")
    try:
        print("Data types going into preprocessor.fit (Classification):")
        print(train_X.dtypes.value_counts())
        preprocessor.fit(train_X, train_y)
        print("Preprocessor fitted successfully.")
        train_X_trans = preprocessor.transform(train_X); test_X_trans = preprocessor.transform(test_X)
        train_X_trans.columns = train_X_trans.columns.astype(str); test_X_trans.columns = test_X_trans.columns.astype(str)
        print(f"Transformed training data shape: {train_X_trans.shape}")
        if train_X_trans.empty and not X.empty:
             print("Warning: Preprocessing resulted in empty training data (Classification).")
    except Exception as e:
        print(f"Error during preprocessing: {e}"); traceback.print_exc()
        # Debugging prints kept
        if isinstance(e, TypeError) and 'argument must be uniformly strings or numbers' in str(e):
             print("\n--- Debugging Mixed Type Error (Classification) ---") # ... (debug prints)
        elif isinstance(e, ValueError) and ('Cannot use' in str(e) or 'fill_value' in str(e)):
             print("\n--- Debugging Imputation Error (Classification) ---")
             try:
                print("  Input column types to preprocessor:")
                print(train_X.info())
             except Exception as debug_e: print(f"  Error during debug imputation check: {debug_e}")
             print("--- End Debugging ---")
        raise ValueError(f"Preprocessing failed: {e}") from e

    # Feature Selection
    print("\nPerforming feature selection using RandomForestClassifier...")
    train_X_selected, test_X_selected = train_X_trans.copy(), test_X_trans.copy()
    if not train_X_trans.empty:
        try:
            selection_estimator = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample', n_jobs=-1, max_depth=10, min_samples_leaf=5)
            max_f = min(100, train_X_trans.shape[1])
            selector = SelectFromModel(estimator=selection_estimator, threshold='median', max_features=max_f, prefit=False)
            selector.fit(train_X_trans, train_y)
            selected_mask = selector.get_support()
            selected_features = train_X_trans.columns[selected_mask]
            if 0 < len(selected_features) < train_X_trans.shape[1]:
                train_X_selected = train_X_trans[selected_features]
                if not test_X_trans.empty: test_X_selected = test_X_trans[selected_features]
                else: test_X_selected = pd.DataFrame(columns=selected_features)
                print(f"Original feature count: {train_X_trans.shape[1]}"); print(f"Selected feature count: {train_X_selected.shape[1]}")
                print(f"Selected features (first 20): {selected_features.tolist()[:20]}...")
            elif len(selected_features) == train_X_trans.shape[1]: print("Feature selection did not remove any features. Using all.")
            else:
                 print("Warning: Feature selection removed all features. Using all original transformed features.")
                 train_X_selected, test_X_selected = train_X_trans.copy(), test_X_trans.copy()
        except Exception as e:
             print(f"Error during feature selection: {e}. Using all features."); traceback.print_exc()
             train_X_selected, test_X_selected = train_X_trans.copy(), test_X_trans.copy()
    else:
        print("Skipping feature selection as transformed training data is empty.")


    # --- Model Training Loop (MODIFIED models dictionary) ---
    models = {
         "GaussianNB": (GaussianNB(), { # <-- Added GaussianNB
             'var_smoothing': logspace(-9, -2, 8) # Variance smoothing parameter
             }, False), # GaussianNB doesn't require explicit scaling in the same way as distance-based models
         "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1, solver='saga'), {'C': [0.01, 0.1, 1, 10, 50]}, True),
         "RandomForest": (RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_jobs=-1), {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}, False),
         "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=42, class_weight='balanced'), {'learning_rate': [0.05, 0.1, 0.2], 'max_iter': [100, 200, 300]}, False),
         "LightGBM": (LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1, verbosity=-1), {'n_estimators': [100, 200, 300], 'learning_rate': [0.05, 0.1, 0.2], 'num_leaves': [20, 31, 50]}, False),
         "MLPClassifier": (MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10), {'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)], 'activation': ['relu', 'tanh'], 'solver': ['adam'], 'alpha': logspace(-5, 0, 5), 'learning_rate': ['constant', 'adaptive']}, True),
         "SVC": (SVC(class_weight='balanced', probability=True, random_state=42), {'C': [0.1, 1, 10, 50], 'kernel': ['rbf', 'linear']}, True),
         "KNN": (KNeighborsClassifier(n_jobs=-1), {'n_neighbors': [5, 9, 15], 'weights':['distance'], 'metric': ['minkowski','cosine']}, True)
     }
    best_model_name=None; best_model_pipeline_obj=None; best_score=-np.inf; model_results=[]; cm_data=[]

    if train_X_selected.empty:
        print("Error: Training data is empty after preprocessing and feature selection. Cannot train models.")
        best_model_details = {"name": "None", "error": "Training data became empty."}
        return best_model_details, pd.DataFrame(), [], None

    for name, (model, param_grid, needs_explicit_scaling) in models.items():
        print(f"\n🧪 Training: {name}")
        model_steps = [];
        if needs_explicit_scaling: model_steps.append(('scaler', StandardScaler()))
        model_steps.append(('model', model))
        model_pipe = Pipeline(model_steps)
        full_pipe = model_pipe
        pipeline_param_grid = {f"model__{k}": v for k, v in param_grid.items()}
        use_smote = False

        # Conditionally add SMOTE using ImbPipeline
        # Note: GaussianNB can sometimes be sensitive to synthetic data from SMOTE,
        # but we keep it consistent with other models for now.
        if min_samples > smote_k_neighbors and name != 'KNN': # Check if SMOTE is possible and not KNN
            use_smote = True
            print(f"  Applying SMOTE (k={smote_k_neighbors}) for {name}.")
            full_pipe = ImbPipeline(steps=[
                ('smote', SMOTE(random_state=42, k_neighbors=smote_k_neighbors)),
                ('model_pipeline', model_pipe) # Nest the original model pipeline
            ])
            # Adjust parameter grid keys for nested pipeline
            pipeline_param_grid = {f"model_pipeline__model__{k}": v for k, v in param_grid.items()}
        elif name == 'KNN':
             print(f"  Skipping SMOTE for KNN model.")
        else: # SMOTE not possible
             print(f"  Skipping SMOTE for {name} as minority samples ({min_samples}) <= k_neighbors ({smote_k_neighbors})")

        searcher = None
        if pipeline_param_grid:
            param_combinations=1;
            for key in pipeline_param_grid:
                 try: full_pipe.get_params()[key.split('__')[0]] ; param_combinations *= len(pipeline_param_grid[key])
                 except KeyError: print(f" Warning: Param '{key}' not found in pipeline for {name}. Skipping."); continue
            actual_n_iter = min(n_iter, param_combinations) if param_combinations > 0 else 0
            if actual_n_iter > 0:
                print(f"  (Using n_iter={actual_n_iter} for RandomizedSearch)")
                searcher = RandomizedSearchCV(full_pipe, pipeline_param_grid, n_iter=actual_n_iter, scoring=scoring_metric, cv=3, n_jobs=-1, random_state=42, error_score='raise')
            else: print(f"  Skipping RandomizedSearch for {name} due to zero valid parameter combinations or empty grid.")

        try:
            if searcher:
                searcher.fit(train_X_selected, train_y)
                best_estimator=searcher.best_estimator_; best_params=searcher.best_params_; cv_score=searcher.best_score_
            else:
                 full_pipe.fit(train_X_selected, train_y)
                 best_estimator=full_pipe; best_params={}; cv_score=None
        except Exception as e: print(f"  Failed to train {name}: {e}"); traceback.print_exc(); continue

        # Evaluate on SELECTED test features
        if test_X_selected.empty:
            print(f"  Warning: Test data is empty for model {name}. Skipping evaluation.")
            acc, f1, bal_acc = np.nan, np.nan, np.nan
        else:
            # Handle potential prediction errors for specific models if needed
            try:
                pred_test = best_estimator.predict(test_X_selected)
                acc = accuracy_score(test_y, pred_test)
                f1 = f1_score(test_y, pred_test, average='weighted' if is_multiclass else 'binary', zero_division=0)
                bal_acc = balanced_accuracy_score(test_y, pred_test)
                try: class_names = [str(c) for c in unique_classes]; print("\nClassification Report (Test Set):\n", classification_report(test_y, pred_test, target_names=class_names, zero_division=0))
                except Exception: print("\nClassification Report (Test Set):\n", classification_report(test_y, pred_test, zero_division=0))
            except Exception as pred_e:
                 print(f"  Error during prediction/evaluation for {name}: {pred_e}")
                 acc, f1, bal_acc = np.nan, np.nan, np.nan # Set metrics to NaN on error

        # Clean parameter names based on whether SMOTE was used
        cleaned_params = {};
        param_prefix_to_remove = 'model_pipeline__model__' if use_smote else 'model__'
        for k, v in best_params.items(): cleaned_params[k.replace(param_prefix_to_remove, '')] = v

        result_entry = {'Model': name, f'CV {scoring_metric}': cv_score if cv_score is not None else 'N/A', 'Test Accuracy': acc, 'Test Balanced Accuracy': bal_acc, 'Test F1': f1, 'Best Params': cleaned_params}
        model_results.append(result_entry); print(f"  📊 Test Metrics -> Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f} | F1 ({'weighted' if is_multiclass else 'binary'}): {f1:.4f}")

        # Determine score for comparison
        current_score_source = "CV"; current_score = cv_score if cv_score is not None and not np.isnan(cv_score) else -np.inf
        if (current_score == -np.inf or pd.isna(current_score)) and not pd.isna(bal_acc): # Use balanced acc as fallback
             current_score = bal_acc; current_score_source = "Test Balanced Acc"
        elif pd.isna(current_score): current_score = -np.inf

        print(f"  Score used for comparison ({current_score_source}): {current_score:.4f}")
        if current_score > best_score:
            print(f"  New best model found: {name} (Score: {current_score:.4f} > Previous best: {best_score:.4f})")
            best_score = current_score; best_model_pipeline_obj = best_estimator; best_model_name = name

    # Save Best Model & Prepare Results
    saved_filename = None; best_model_details = {}; final_cm_data = []
    if best_model_pipeline_obj is not None:
        try:
            if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
            filename_base = f"model_class_{best_model_name.replace(' ', '')}_{uuid.uuid4().hex[:8]}"
            saved_filename = f"{filename_base}.pkl"
            save_path = os.path.join(MODEL_SAVE_DIR, saved_filename)
            joblib.dump(best_model_pipeline_obj, save_path)
            print(f"\n✅ Best model '{best_model_name}' saved to: {save_path}")
            best_model_results_entry = next((item for item in model_results if item['Model'] == best_model_name), None)
            best_model_details = {"name": best_model_name, "pipeline_steps": str(best_model_pipeline_obj.steps), "best_params": best_model_results_entry['Best Params'] if best_model_results_entry else {}}
            if not test_X_selected.empty:
                 # Handle potential prediction error for the final best model CM
                 try:
                     final_preds = best_model_pipeline_obj.predict(test_X_selected)
                     cm = confusion_matrix(test_y, final_preds, labels=unique_classes)
                     print("\n📊 Confusion Matrix (Best Model):\n", cm); final_cm_data = cm.tolist()
                 except Exception as cm_pred_e:
                      print(f"  Error during prediction for final CM: {cm_pred_e}")
                      final_cm_data = [] # Set empty CM on error
            else:
                 print("\n📊 Confusion Matrix (Best Model): Skipped (Test data empty)")
                 final_cm_data = []
        except Exception as e: print(f"Error saving model {best_model_name}: {e}"); traceback.print_exc(); saved_filename = None; best_model_details = {"name": best_model_name, "error": "Failed to save model."}; final_cm_data = []
    else: print("\n⚠️ No best model found."); best_model_details = {"name": "None", "error": "No suitable model was found."}

    if not model_results: print("Warning: No models trained."); return best_model_details, pd.DataFrame(), final_cm_data, saved_filename
    # Ensure results_df is created even if some models failed
    results_df = pd.DataFrame(model_results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Test Balanced Accuracy', ascending=False).reset_index(drop=True)
    print(f"\nClassification complete. Best Model: {best_model_name}")
    return best_model_details, results_df, final_cm_data, saved_filename
