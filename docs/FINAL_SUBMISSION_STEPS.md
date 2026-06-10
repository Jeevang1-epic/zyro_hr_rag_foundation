# Final Submission Steps

1. Deploy Streamlit app.
2. Copy public Streamlit URL.
3. Run one RAG query with LangSmith tracing enabled.
4. Open LangSmith trace.
5. Share trace publicly.
6. Copy public LangSmith URL.
7. Set environment variables:

Windows PowerShell:

```powershell
$env:STREAMLIT_APP_URL="https://your-app.streamlit.app"
$env:LANGSMITH_TRACE_URL="https://smith.langchain.com/public/..."
```

Then run:

```powershell
python scripts/run_live_submission.py
```

8. Validate `outputs/submission.csv`.
9. Upload `outputs/submission.csv` to Kaggle only if validation says `SUBMISSION READY FOR KAGGLE UPLOAD`.

Run final validation with:

```powershell
python scripts/validate_final_submission.py
```

Deployment note: `app/streamlit_app.py` uses relative repository paths and expects the HR policy PDFs to be available at `datasets/zyro-dynamics-hr-corpus/` in the deployed environment. Do not commit the dataset unless the competition rules/license explicitly allow it.
