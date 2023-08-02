FROM python:3.10-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install PyTorch
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Set the entrypoint command to run train.py
ENTRYPOINT ["python","train.py"]