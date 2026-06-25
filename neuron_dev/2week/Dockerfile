# KB 부동산 RAG 챗봇 — 완전 독립 Python 이미지
# unsloth 이미지와 무관. 순수 python:3.11-slim 위에 필요한 것만 설치.
FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 설치(레이어 캐시)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 앱 본체
COPY app.py /app/app.py

EXPOSE 8501

# 9p/바인드마운트 파일와처 폭주 방지: fileWatcherType none
# 리버스 프록시(NPM) 뒤에서 동작: CORS/XSRF off, 0.0.0.0 바인딩
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.fileWatcherType=none"]
