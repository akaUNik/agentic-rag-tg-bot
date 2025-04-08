FROM python:3.11-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install pip --upgrade
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the bot's code to the working directory
COPY . .

# Run bot.py when the container launches
CMD ["python", "bot.py"]