# Use the official PyTorch base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files to the container
COPY requirements.txt .
COPY your_custom_model.py .
COPY pytorch_model.bin .

# Install the required dependencies
RUN pip install -r requirements.txt

# Expose any necessary ports
EXPOSE 8000

# Set the entry point command
CMD ["python", "your_custom_model.py"]
