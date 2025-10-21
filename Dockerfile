FROM python:3.13-alpine

# Set the working directory in the container
WORKDIR /app

RUN apk add --no-cache libmagic

# Copy the requirements file into the container at /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install -U pip --root-user-action=ignore
RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Copy the rest of the application code into the container at /app
COPY adk-bot/ /app

# Define the command to run your application
CMD ["python", "main.py"]