<div align="center">

# 🧠 WellbeingIQ
### AI-Powered Student Wellness Prediction & Guidance System

**Understand digital habits. Explore wellbeing patterns. Get practical, non-clinical guidance.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-8A2BE2)](https://shap.readthedocs.io/)
[![GenAI](https://img.shields.io/badge/GenAI-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Capstone Project 2 (CP2)** · Postgraduate Program in Data Science & Analytics with GenAI  
**Imarticus Learning · Batch PGA-56 · Hyderabad, India**

</div>

---

## 🌱 Overview

**WellbeingIQ** is an educational prototype that explores how students' digital habits, sleep routines, physical activity, academic pressure, and social support relate to broad student-wellness categories.

The project brings together a complete analytics and machine-learning workflow:

1. 🧹 Prepare and clean a synthetic student dataset.
2. 🔎 Explore patterns and develop data-backed observations.
3. 🤖 Train and evaluate machine-learning classification models.
4. 🧩 Explain individual predictions using SHAP.
5. 💬 Convert model explanations into practical habit suggestions with an optional Gemini-powered coach.
6. 🖥️ Present the workflow through an interactive Streamlit application.

> ⚠️ **Important:** WellbeingIQ is an academic prototype, not a medical or clinical product. It does not diagnose, assess, or treat any health condition. Predictions are estimates from a synthetic dataset and must not be used for real-world decisions about students.

## ✨ Key Features

| Feature | What it does |
|---|---|
| 📊 Wellness Check | Accepts habit-related inputs and returns one of three model categories: `Good`, `Moderate`, or `At-Risk`. |
| 🔍 Explainable Predictions | Uses SHAP contributions to show factors that influenced a model prediction. |
| 🎛️ What-If Simulator | Lets users explore how changing selected habits changes the model output. |
| 💬 Wellness Coach | Provides general, habit-focused suggestions. Gemini is optional; a built-in fallback is available. |
| 🛡️ Uncertainty Note | Flags lower-confidence predictions instead of presenting every estimate as certain. |
| 📈 Analytics Workflow | Includes data preparation, EDA, model evaluation, explainability, and GenAI notebooks. |

## 🧪 Model Snapshot

The supplied `model_card.json` records the following test-set results for the saved Logistic Regression pipeline:

| Metric | Recorded result |
|---|---:|
| Accuracy | 0.630 |
| Macro Precision | 0.623 |
| Macro Recall | 0.670 |
| Macro F1-score | 0.635 |
| At-Risk Recall | 0.767 |

These are results from the project's supplied synthetic dataset and test split. They are not evidence of clinical validity or real-world performance.

## 🧰 Tech Stack

- **Language:** Python
- **Data handling:** pandas, NumPy
- **Visual analysis:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn
- **Explainable AI:** SHAP, LIME
- **GenAI:** Google Gemini API (optional)
- **Interactive application:** Streamlit
- **Model persistence:** joblib
- **Development:** Jupyter Notebook

## 🗂️ Repository Structure

```text
WellbeingIQ/
├── app.py
├── wellbeing_features.py
├── wellbeing_explain.py
├── wellbeing_coach.py
├── wellbeingiq_model.joblib
├── model_card.json
├── explain_config.json
├── requirements.txt
│
├── 01_Data_Cleaning_Preparation.ipynb
├── 02_EDA_Business_Insights.ipynb
├── 03_Modeling_Evaluation.ipynb
├── 04_Explainability_SHAP.ipynb
├── 05_GenAI_Coach_Gemini.ipynb
│
├── WellbeingIQ_Synthetic_Student_Dataset.csv
├── WellbeingIQ_Synthetic_Student_Dataset_Cleaned.xlsx
├── WellbeingIQ_cleaned.csv
├── cleaning_log.csv
├── global_importance.csv
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
│
├── Streamlit Dashboard Screenshot/
│   ├── Dashboard 1 Wellness Check.png
│   └── Dashboard 2 What-If Simulator.png
│
├── CP2- Proposal- WellbeingIQ-AI-Powered Student Wellness Prediction and Guidance System.pdf
├── WellbeingIQ-CP2-Report.pdf
├── WellbeingIQ-CP2-PPT.pptx
├── README.md
├── LICENSE
└── .gitignore
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mrgulshan87/WellbeingIQ-AI-Powered-Student-Wellness-Prediction-and-Guidance-System.git
cd WellbeingIQ-AI-Powered-Student-Wellness-Prediction-and-Guidance-System
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application should open in your browser. Keep the project files together in the repository root because the app loads the saved model, model card, configuration, and training data using relative paths.

### 5. (Optional) Enable Gemini

The coach includes a built-in fallback, so the application can be explored without an API key. To enable Gemini, set your key as an environment variable. Do not commit API keys to GitHub.

**PowerShell (current terminal session):**

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
streamlit run app.py
```

## 📓 Notebook Workflow

Run the notebooks in this order:

| Order | Notebook | Purpose |
|---:|---|---|
| 01 | `01_Data_Cleaning_Preparation.ipynb` | Data quality checks, cleaning, and preparation |
| 02 | `02_EDA_Business_Insights.ipynb` | Exploratory analysis and observations |
| 03 | `03_Modeling_Evaluation.ipynb` | Model training, comparison, evaluation, and saved artifacts |
| 04 | `04_Explainability_SHAP.ipynb` | SHAP explanations and global feature importance |
| 05 | `05_GenAI_Coach_Gemini.ipynb` | GenAI coach workflow and safe fallback behaviour |

Run notebooks from the repository root so relative file paths resolve correctly.

## 🖥️ Application Screenshots

### 🌿 Wellness Check
![WellbeingIQ Wellness Check](Streamlit%20Dashboard%20Screenshot/Dashboard%201%20Wellness%20Check.png)

### 🎛️ What-If Simulator
![WellbeingIQ What-If Simulator](Streamlit%20Dashboard%20Screenshot/Dashboard%202%20What-If%20Simulator.png)

## 🔐 Privacy, Safety & Responsible Use

- The supplied project dataset is labelled synthetic.
- Do not add real student names, contact details, IDs, or other personally identifying information.
- Do not commit API keys, credentials, `.env` files, or private configuration.
- The model output is a broad educational category, not a diagnosis.
- SHAP values explain model behaviour; they do not establish cause and effect.
- Use the app for learning and demonstration, not for screening, admissions, counselling, or other high-impact decisions.

## 🔮 Future Enhancements

- Validate the workflow on appropriately governed, representative real-world data before considering any real-world use.
- Add stronger model calibration and clearer uncertainty reporting.
- Expand accessibility and multilingual support.
- Add automated tests and a reproducible experiment configuration.
- Deploy the Streamlit application with secure secret management.

## 📄 Project Documents

The repository includes the CP2 proposal, final report, and presentation deck for project review and academic reference.

## 👨‍💻 Author

**Gulshan Kumar Kushwaha**  
Data Analyst | BI & GenAI Analytics

- GitHub: [@mrgulshan87](https://github.com/mrgulshan87)
- LinkedIn: [Gulshan Kumar Kushwaha](https://www.linkedin.com/in/gulshan-kumar-kushwaha-b62282214/)

---

<div align="center">

**Built with curiosity, data, and a focus on responsible AI. 💚**

</div>
