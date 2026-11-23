# HealthHub AI — Streamlit deployment prototype

This is a **Streamlit adaptation** of the seven-module Jupyter notebook. It is not a byte-for-byte port: it retains the core demonstration workflows, while simplifying the original HealthWatch NLTK tokenization and some graphing, knowledge-acquisition, and logging features. Test every module before publishing or using for coursework demonstrations.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Publish

1. Create a GitHub repository and upload `app.py`, `healthhero_content.py`, and `requirements.txt`.
2. Visit https://share.streamlit.io/ and connect your GitHub account.
3. Choose the repository and `app.py` as the entrypoint, then deploy.
4. Test all seven modules on the public URL.

## Files needed

- **DiaBot**: Upload the original `diabetesRiskFactors.pdf.pdf` through the DiaBot page, or add it to the project **only if you have the right to redistribute it**. The PDF is not included.
- **MedCost AI**: Upload `medical_cost.csv` on the MedCost page. The dataset is not included.
- **HealthWatch AI**: Needs outbound access to Project Gutenberg and may be subject to site availability or rate limits.

## Safety and privacy

Educational demonstration only. Do not enter real identifiable patient data or upload private records. Streamlit hosting is not automatically HIPAA compliant. State is per-browser session; CareFlow and VitalGuard data are simulated. No clinical validity is claimed.

## HealthSentiment AI
HealthWatch has been replaced with HealthSentiment AI. Upload the original `drugsComTest_raw.csv` and `patient_feedback_dataset.xlsx` files on the HealthSentiment page. The uploaded data are processed temporarily in the current session, and results can be downloaded as CSV. Use anonymized or synthetic data only.
