# Isharati on a Hugging Face Docker Space: the React landing page (web/dist) and the FastAPI app
# (isharati.app.server) with its data fetched from the private dataset at start-up (isharati.hub), models through
# OpenRouter (then NVIDIA NIM / Groq when their keys are set), question embeddings by HF inference.

FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 PORT=7860 \
    ISHARATI_DATA_DIR=/home/user/data ISHARATI_OUT_DIR=/home/user/data/out \
    ISHARATI_LLM=openrouter ISHARATI_EMBED=hf ISHARATI_LANGS=en,ar,tr,ur ISHARATI_URDU_WSLP=1 \
    ISHARATI_WEB_DIST=/home/user/app/web/dist \
    PYTHONPATH=/home/user/app/src
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user src ./src
COPY --chown=user web/dist ./web/dist

EXPOSE 7860
CMD ["python", "-m", "isharati.app.server"]
