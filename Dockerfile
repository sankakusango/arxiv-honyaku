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
RUN bash install_texlive.sh 2023 https://texlive.info/historic/systems/texlive/2023/tlnet-final
RUN bash install_texlive.sh 2025 https://texlive.info/historic/systems/texlive/2025/tlnet-final

# Some arXiv sources include EPS assets; epstopdf needs Ghostscript (`gs`).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ghostscript \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV PATH="/opt/texlive/2025/bin/x86_64-linux:${PATH}"

CMD ["sleep", "infinity"]
