FROM python:3.11-slim
WORKDIR /yolo_app
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    # Additional OpenCV GUI dependencies
    libgtk-3-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Camera and video dependencies
    v4l-utils \
    # Cleanup to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application code into the container
COPY . .
# Expose port 5000 for the Flask app
EXPOSE 5000
# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
# Define the command to run the Flask app
CMD ["python", "detection_flask.py"]