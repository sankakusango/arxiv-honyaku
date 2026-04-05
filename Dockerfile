FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        perl \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/install_texlive.sh .

# Keep one TeX Live year per Docker layer so adding a new year can reuse cache
# for already-installed years.
RUN bash install_texlive.sh 2023 http://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2023/tlnet-final scheme-full
RUN bash install_texlive.sh 2025 http://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final scheme-full

ENV PATH="/opt/texlive/2025/bin/x86_64-linux:${PATH}"

COPY . /app/
RUN mkdir -p /app/runs
EXPOSE 8000

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install ".[web]"

CMD ["uvicorn", "arxiv_honyaku.interfaces.web:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
