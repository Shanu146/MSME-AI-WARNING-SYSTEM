
JIVAN Prototype - README
Files created under /mnt/data:
 - jivan_synthetic_dataset.csv  (synthetic sample data)
 - jivan_model.pkl              (trained RandomForest model)
 - app.py                       (Streamlit app to run)
 - confusion_matrix.png
 - risk_distribution.png

How to run locally:
1) Install Python packages:
   pip install streamlit pandas scikit-learn matplotlib joblib

2) Move the three files (app.py, jivan_model.pkl, jivan_synthetic_dataset.csv) into the same folder,
   then run:
   streamlit run app.py

What to test in the app:
 - Upload your own CSV (must include required columns) or use the sample data.
 - Inspect top flagged MSMEs, download results, and view the predicted distribution.

Next steps I can help with (choose one):
 - Improve features and model (e.g., time-series features from monthly consumption)
 - Add SHAP explanation and feature-level scores per MSME
 - Secure deployment (Docker + simple backend)
 - Integrate real proxy datasets (electricity + Udyam metadata)

