FROM python:3.11-slim

WORKDIR /app

COPY ./src/ /app

COPY ./pyproject.toml /app/pyproject.toml
RUN pip install /app/.

CMD ["bash", "entrypoint.sh"]