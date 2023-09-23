FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

COPY ../requirements-noplot.txt /tmp/pip-tmp/requirements.txt

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install git gcc unzip make -y \
    && pip3 install --disable-pip-version-check --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp
