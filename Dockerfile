# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and any other necessary files (e.g., README, LICENSE)
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Install dependencies from the pyproject.toml file
RUN pip install --upgrade pip setuptools wheel
RUN pip install .

ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Install the necessary packages for the FastAPI app
RUN pip install fastapi "uvicorn[standard]" gunicorn transformers accelerate huggingface_hub hf-transfer "jinja2>=3.1.0"

# Copy the entire project code into the container
COPY . /app

# Copy the serve script into the container
COPY serve /usr/local/bin/serve

# Make the serve script executable
RUN chmod +x /usr/local/bin/serve

# Set environment variable to determine the device (cuda or cpu)
ENV env=prod

# Expose the port that the FastAPI app will run on
EXPOSE 8080

# Set the entrypoint for SageMaker to the serve script
ENTRYPOINT ["serve"]
