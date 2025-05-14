# Auto_Machine_Learning_Program
This is a website app that allow client submit their own file and automatic doing the machine learning and give the prediction pkl and metrics.

---

![automl_summary](https://github.com/user-attachments/assets/9a6beb14-ba5c-47d5-90e4-438aea0019f6)


# Using the Auto Machine Learning Program

This guide provides step-by-step instructions on how to use the **Auto Machine Learning Program** to upload datasets and perform automatic machine learning tasks.

---

## Prerequisites

Before you begin, ensure you have the following installed:

- [Python](https://www.python.org/) (version 3.8 or higher)
- [Docker](https://www.docker.com/) (optional, for containerized deployment)
- A modern web browser (e.g., Chrome, Firefox)

---

## Step 1: Clone the Repository

First, download the code from the GitHub repository.

1. Open your terminal (Command Prompt, PowerShell, or any terminal on your system).
2. Run the following commands:
   ```bash
   git clone https://github.com/jimmy-chen-1/Auto_Machine_Learning_Program.git
   cd Auto_Machine_Learning_Program
   ```

---

## Step 2: Set Up the Environment

### Option 1: Using a Virtual Environment

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Docker (Optional)

If you prefer containerized deployment, you can set up the environment using Docker:

1. Build the Docker image:
   ```bash
   docker build -t auto-ml-program .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 auto-ml-program
   ```

---

## Step 3: Run the Application

If you set up the virtual environment:

1. Run the application using:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

If you used Docker, the application will automatically be available at:
```
http://localhost:5000
```

---

## Step 4: Upload Your Dataset

1. On the main webpage, click the **Upload File** button.
2. Select your dataset in **CSV format**.
3. Once uploaded, the application will:
   - Clean and preprocess the data.
   - Train and test a machine learning model based on your dataset.
   - Generate:
     - Prediction results.
     - A serialized model (`.pkl` file).
     - Performance metrics.

---

## Step 5: Review the Results

1. After the process is complete, the webpage will display:
   - **Performance Metrics**: Key details about the model's performance.
   - **Download Links**:
     - The trained model as a `.pkl` file.
     - Prediction results for offline analysis.

2. You can review the results directly on the webpage or download the files for further use.

---

## Step 6: Customizing the Code (Optional)

If you'd like to modify the code to add new features or customize the machine learning pipeline:

1. Open the repository directory in your favorite code editor (e.g., VS Code, PyCharm).
2. Edit the relevant files:
   - `app.py`: Backend logic for the application.
   - Frontend HTML files for UI customization.
3. After making changes, test your updates by re-running:
   ```bash
   python app.py
   ```
   Or rebuild the Docker container:
   ```bash
   docker build -t auto-ml-program .
   ```

---

## Troubleshooting

Here are solutions to common issues:

1. **Python version is too old**:
   - Install Python 3.8 or higher from [python.org](https://www.python.org/).

2. **Dependencies not installed**:
   - Ensure you ran:
     ```bash
     pip install -r requirements.txt
     ```

3. **Port already in use**:
   - If `http://localhost:5000` doesn't open, check if another application is using port 5000.
   - You can change the port in `app.py` by modifying:
     ```python
     app.run(port=5000)
     ```
     Replace `5000` with another available port, e.g., `8080`.

---

Follow these steps, and you'll be able to use the **Auto Machine Learning Program** successfully!
