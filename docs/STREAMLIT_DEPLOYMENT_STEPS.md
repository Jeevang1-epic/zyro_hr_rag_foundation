# Streamlit Deployment Steps

1. Confirm no restricted files are committed: `datasets/`, `outputs/`, `.env`, `kaggle.json`, zip files, and API keys must stay out of git.
2. Push the safe repository code to GitHub only after approval.
3. Open Streamlit Community Cloud and create a new app.
4. Select the GitHub repository, branch, and `app/streamlit_app.py`.
5. Add secrets in the Streamlit secrets UI if you later enable an external LLM. Do not commit secrets to git.
6. Deploy the app and verify that it can load the HR PDF corpus.
7. Copy the public `https://...streamlit.app` URL.
8. Set it locally as `STREAMLIT_APP_URL` before regenerating `outputs/submission.csv`.

Current status: no real public Streamlit URL is available in this local run, so the generated CSV remains `DRAFT_ONLY`.

Deployment data note: `app/streamlit_app.py` expects the HR PDFs at `datasets/zyro-dynamics-hr-corpus/` relative to the repository root. Do not commit the PDFs unless the competition rules/license explicitly allow it.
