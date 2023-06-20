# Use the official PyTorch base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set the working directory in the container
WORKDIR /home/vy/Question-Answer-Generation

# Copy the necessary files to the container
COPY requirements.txt .
COPY requirement-api.txt .

# Install the required dependencies
RUN pip install -r requirements.txt
RUN pip install -r requirement-api.txt

COPY src/api .
COPY checkpoints .

RUN cd ./src/api/
# Set the entry point command
CMD ["uvicorn", "main:app", "--reload"]
