# LangSmith Trace Steps

1. Set the LangSmith or LangChain API key in your local shell or deployment environment. Do not commit it.
2. Set `LANGCHAIN_TRACING_V2=true`.
3. Set `LANGCHAIN_PROJECT=zyro-rag-challenge`.
4. Run the final RAG answer generation command after tracing is enabled.
5. Open `https://smith.langchain.com` and sign in.
6. Open the `zyro-rag-challenge` project.
7. Open a completed trace, click Share, enable the public link, and copy the URL.
8. Set it locally as `LANGSMITH_TRACE_URL` before regenerating `outputs/submission.csv`.

Current status: no LangSmith key or public trace URL is available in this local run, so the generated CSV remains `DRAFT_ONLY`.
