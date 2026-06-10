# Medical_Report_Analysis 🧠🩻  
NLP Model for Extracting Organ Presence Status from Medical Reports

This project is part of a broader **Medical Report Analysis System** designed to verify organ presence by comparing:

- **What is written in the clinical report (NLP)**  
  vs  
- **What is detected in MRI/CT scans by the Computer Vision model**

The goal is to detect any **inconsistency** between text and imaging that may indicate human error, reporting mistakes, or critical mismatch.

---

## 🧠 Project Description

Given an English **medical abdominal imaging report** (MRI/CT), the NLP model predicts the **presence status** of four abdominal organs:

- `left_kidney` → `present` / `missing` / `removed`  
- `right_kidney` → `present` / `missing` / `removed`  
- `liver` → `present` / `missing` / `removed`  
- `spleen` → `present` / `missing` / `removed`  

Each report may describe multiple organs, and the model performs **multi-output classification** to produce one prediction per organ.

---

## 🔧 Project Structure

```bash
Medical_Report_Analysis/
├─ data/
│  ├─ raw/
│  │  └─ nlp_medical_reports_multiorgan.csv   # synthetic medical-style reports (training data)
│  └─ processed/                              # (optional) cleaned/tokenized data
├─ models/
│  └─ nlp_presence_model.joblib               # saved trained NLP model
├─ src/
│  ├─ train_nlp_model.py                      # model training & evaluation pipeline
│  └─ predict_nlp.py                          # inference on new medical reports
├─ notebooks/
│  └─ exploration.ipynb                       # optional EDA / experiments / testing
└─ README.md
