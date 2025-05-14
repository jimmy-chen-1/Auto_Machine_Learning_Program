# main.py (Final Version with English Comments/Output)

import pandas as pd
import numpy as np
import io
import traceback
import os
import uuid # Although filename generation moved to ml_logic, keep for potential use
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict, Any
import uvicorn

# Import ML logic functions and constants
from ml_logic import (
    create_preprocessor,
    auto_train_clean_data,
    auto_train_classification,
    MODEL_SAVE_DIR # Import save directory constant
)

app = FastAPI(
    title="AutoML Web App",
    description="Upload a CSV, select target and task type, get best model results and download the model."
)

# Setup Jinja2 Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
# Check if template directory exists, provide helpful error if not
if not os.path.isdir(TEMPLATE_DIR):
    print(f"ERROR: Template directory not found at {TEMPLATE_DIR}")
    # Optionally raise an error or exit if templates are essential
    # raise RuntimeError(f"Template directory not found: {TEMPLATE_DIR}")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


# Ensure model save directory exists on startup
# Check if MODEL_SAVE_DIR is defined and is a valid path before creating
if MODEL_SAVE_DIR and isinstance(MODEL_SAVE_DIR, str):
    try:
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        print(f"Model save directory ensured at: {os.path.abspath(MODEL_SAVE_DIR)}")
    except OSError as e:
        print(f"ERROR: Could not create model save directory '{MODEL_SAVE_DIR}': {e}")
        # Decide how to handle this - maybe raise error or disable saving?
        MODEL_SAVE_DIR = None # Disable saving if directory fails
else:
    print("Warning: MODEL_SAVE_DIR not configured correctly in ml_logic.py. Model saving disabled.")
    MODEL_SAVE_DIR = None


# Endpoint to Serve HTML Form
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page with the upload form."""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template 'index.html': {e}")
        traceback.print_exc()
        # Provide a fallback HTML response or raise a specific HTTPException
        return HTMLResponse(content="<html><body><h1>Error</h1><p>Could not load the application interface. Please check server logs.</p></body></html>", status_code=500)


# POST Endpoint for AutoML Processing
@app.post("/automl/", response_model=Dict[str, Any])
async def run_automl(
    # Function arguments remain the same
    file: UploadFile = File(..., description="CSV file to process"),
    target_col: str = Form(..., description="Name of the target variable column"),
    task_type: str = Form(..., description="Type of task: 'regression' or 'classifier'", enum=["regression", "classifier"]),
    date_cols_str: Optional[str] = Form(None, description="Comma-separated list of date/year column names (optional)")
):
    print(f"\n--- Received AutoML Request ---")
    print(f"Filename: {file.filename}, Task: {task_type}, Target: {target_col}")
    # --- File Reading and Initial Checks ---
    try:
        contents = await file.read()
        # Check if file is empty
        if not contents:
             raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        buffer = io.BytesIO(contents)
        # Try reading CSV with basic error handling
        try:
             df = pd.read_csv(buffer)
        except pd.errors.EmptyDataError:
             raise HTTPException(status_code=400, detail="CSV file is empty or invalid.")
        except pd.errors.ParserError:
            raise HTTPException(status_code=400, detail="Could not parse CSV file. Ensure it is correctly formatted.")
        except Exception as e: # Catch other potential pandas errors
            raise HTTPException(status_code=400, detail=f"Error reading CSV content: {e}")
        finally:
             buffer.close() # Ensure buffer is closed

        print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        if df.empty:
             raise HTTPException(status_code=400, detail="CSV contains no data after reading headers.")

    except Exception as e: # Catch errors related to file reading itself
        print(f"Error reading uploaded file: {e}"); traceback.print_exc()
        # Use detail from exception if it's already an HTTPException
        detail = e.detail if isinstance(e, HTTPException) else f"Error processing uploaded file: {e}"
        status_code = e.status_code if isinstance(e, HTTPException) else 400
        raise HTTPException(status_code=status_code, detail=detail)

    # Validate target column
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        valid_cols_preview = list(df.columns)[:10] # Show first few columns
        detail = f"Target column '{target_col}' not found. Available columns start with: {valid_cols_preview}{'...' if len(df.columns) > 10 else ''}"
        raise HTTPException(status_code=400, detail=detail)

    # Parse and validate date columns
    user_date_cols = None
    if date_cols_str:
        user_date_cols = [col.strip() for col in date_cols_str.split(',') if col.strip()]
        missing_date_cols = [col for col in user_date_cols if col not in df.columns]
        if missing_date_cols:
             print(f"Missing user-specified date columns: {missing_date_cols}")
             raise HTTPException(status_code=400, detail=f"Provided date column(s) not found in CSV: {missing_date_cols}")
    print(f"User-specified date columns: {user_date_cols}")

    # Drop rows with NaN in target column
    initial_rows = len(df)
    df.dropna(subset=[target_col], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0: print(f"Dropped {rows_dropped} rows with missing target ('{target_col}') values.")
    if len(df) == 0: raise HTTPException(status_code=400, detail=f"No valid data remaining after dropping rows with missing target column '{target_col}'.")
    if len(df) < 10: print(f"Warning: Very few data points ({len(df)}) remaining after dropping NAs in target.")
    # --- End Initial Checks ---

    try:
        # 1. Create the *unfitted* Preprocessor object
        # This step also classifies features internally
        preprocessor, cols_to_drop_lists = create_preprocessor(df, target_col, user_date_cols)

        # 2. Prepare final X, y by dropping identified columns
        cols_to_drop = set(col for sublist in cols_to_drop_lists for col in sublist if col in df.columns and col != target_col)
        print(f"Columns identified by classifier to drop: {list(cols_to_drop)}")
        df_cleaned = df.drop(columns=list(cols_to_drop), errors='ignore')
        X = df_cleaned.drop(columns=[target_col])
        y = df_cleaned[target_col]
        print(f"Features shape for training (X): {X.shape}")
        if X.empty or X.shape[1] == 0: raise ValueError("No feature columns remaining after initial dropping/preparation.")

        # 3. Call appropriate training function (fitting now happens inside)
        results = {}
        cm_data = None
        saved_model_filename = None

        print(f"\n--- Starting {task_type.capitalize()} Task ---")
        if task_type == "regression":
            if not pd.api.types.is_numeric_dtype(y): raise HTTPException(status_code=400, detail=f"Target column '{target_col}' must be numeric for regression.")
            best_model_details, results_df, saved_model_filename = auto_train_clean_data(
                X, y, preprocessor, search_type='random', n_iter=20
            )
            results["model_comparison"] = results_df.round(4).to_dict(orient='records') if not results_df.empty else []
            results["best_model"] = best_model_details

        elif task_type == "classifier":
             if pd.api.types.is_numeric_dtype(y) and y.nunique() > 50:
                 print(f"Warning: Target column '{target_col}' appears numeric with {y.nunique()} unique values. Treating as classification.")
                 if np.all(y == y.astype(int)): y = y.astype(int)
             try:
                 y = y.astype('category') # Ensure consistent type for classification
             except Exception as e:
                 raise HTTPException(status_code=400, detail=f"Could not convert target column '{target_col}' to categorical type: {e}")

             best_model_details, results_df, cm_data, saved_model_filename = auto_train_classification(
                X, y, preprocessor, search_type='random', n_iter=20
             )
             results["model_comparison"] = results_df.round(4).to_dict(orient='records') if not results_df.empty else []
             results["best_model"] = best_model_details
             results["confusion_matrix"] = cm_data

        print(f"--- {task_type.capitalize()} Task Completed ---")

        # 4. Prepare final JSON response
        final_response = {
             "message": "AutoML process completed successfully.",
             "task_type": task_type,
             "target_column": target_col,
             "download_filename": saved_model_filename, # Include filename for download link
             **results
         }
        return JSONResponse(content=final_response)

    # --- Error Handling ---
    except HTTPException as http_exc:
        # Re-raise specific user-facing errors
        raise http_exc
    except ValueError as ve:
        # Handle data validation or processing errors
        print(f"ValueError during processing: {ve}"); traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Data validation or processing error: {ve}")
    except MemoryError:
        print("MemoryError encountered during processing."); traceback.print_exc()
        raise HTTPException(status_code=500, detail="Processing failed due to insufficient memory. Try with a smaller dataset or more resources.")
    except ImportError as ie:
        # Handle missing optional dependencies if fallbacks weren't sufficient
         print(f"ImportError: {ie}"); traceback.print_exc()
         raise HTTPException(status_code=500, detail=f"Server configuration error: Missing required library ({ie.name}). Please check server setup.")
    except Exception as e:
        # Catch-all for other unexpected errors during ML processing
        print(f"Unexpected error during ML processing: {type(e).__name__}: {e}"); traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during the AutoML process: {type(e).__name__}. Check server logs for details.")


# --- Endpoint to Download Saved Models ---
@app.get("/download_model/{filename}", response_class=FileResponse)
async def download_model(filename: str):
    """Provides the saved model file (.pkl) for download."""
    # Basic input validation/sanitization
    if not filename or ".." in filename or filename.startswith("/") or not filename.endswith(".pkl"):
        print(f"Download attempt with invalid filename: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename format.")

    if MODEL_SAVE_DIR is None:
         raise HTTPException(status_code=500, detail="Model saving is disabled due to directory configuration error.")

    file_path = os.path.join(MODEL_SAVE_DIR, filename)
    print(f"Attempting to provide download for: {file_path}")

    if not os.path.isfile(file_path): # More specific check for file existence
        print(f"Model file not found at path: {file_path}")
        raise HTTPException(status_code=404, detail="Model file not found. It might have expired or failed to save.")

    # Suggest a user-friendly download name
    try:
        # Extracts model type and name if possible from filename like 'model_class_LGBM_uuid.pkl'
        parts = filename.replace('.pkl','').split('_')
        if len(parts) >= 3:
            # e.g., best_classifier_LightGBM.pkl
            download_suggested_name = f"best_{parts[1]}_{parts[2]}.pkl"
        else:
            download_suggested_name = f"best_model_{filename}" # Fallback includes original filename
    except Exception:
        download_suggested_name = "best_model.pkl" # Generic fallback

    return FileResponse(
        path=file_path,
        media_type='application/octet-stream', # Standard MIME type for downloads
        filename=download_suggested_name # Suggested filename for the user
    )

# --- Run instruction ---
if __name__ == "__main__":
    print("--- AutoML FastAPI App ---")
    print("This script defines the API endpoints.")
    print("To run the application, use the command in your terminal:")
    print("uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    # Example using uvicorn library directly (less common for deployment)
    # uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)