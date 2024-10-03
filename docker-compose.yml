# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY src/electrical_load_dashboard/requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/electrical_load_dashboard/ .

# Expose the port the app runs on
EXPOSE 8050

# Command to run the application
CMD ["python", "app.py"]
