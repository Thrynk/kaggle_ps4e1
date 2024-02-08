FROM python:3.11-bookworm AS base
LABEL name="AUNTIE"
LABEL version="1.1"

# updating the base system
RUN apt-get update -y

# installing depndencies (poetry first then requirements.txt)
RUN if [ -f "pyproject.toml" ] && [ -f "poetry.lock" ]; then \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi;

# Cleaning
RUN apt-get autoremove -yqq --purge && \
    rm -rf /var/cache/apt && \
    rm -rf ~/.cache/pip && \
    apt-get clean

# copying base DS code
RUN mkdir src
COPY /src /src/