

# Use a newer version of Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=300 -r requirements.txt


# Copy files
COPY ./fastapi_app ./fastapi_app
COPY ./streamlit_app ./streamlit_app





# Create log directory
RUN mkdir -p /var/log

EXPOSE 8501 8000

# CMD bash -c "\
#   uvicorn fastapi_app.main:app --host 0.0.0.0 --port=8000 > /var/log/fastapi.log 2>&1 & \
#    streamlit run streamlit_app/app.py --server.port=8501 --server.enableCORS=false > /var/log/streamlit.log 2>&1 & \
#    wait"

CMD ["bash", "-c", "uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app/app.py --server.port=8501 --server.enableCORS=false & wait"]
