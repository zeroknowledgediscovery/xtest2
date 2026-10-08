FROM python:3.13-slim
WORKDIR /workspace
RUN python3 -m pip install --no-cache-dir numpy==2.2.6
COPY data/ /workspace/data/
RUN mkdir -p /workspace/output /solution /tests /logs/verifier && \
    useradd --create-home agent && chown -R agent:agent /workspace /logs
USER agent
