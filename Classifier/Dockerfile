FROM python:3.9
WORKDIR /app
COPY . /app
RUN ls -R /app
RUN pip install -e ./app
EXPOSE 80
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "80"]
