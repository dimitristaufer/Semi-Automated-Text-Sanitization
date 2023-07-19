### Screen Stuff ###

# Start Screen:
# screen -S inferenceScreen

# Detach:
# Ctrl+A then Ctrl+D

# List screens:
# screen -ls

# Reattacht:
# screen -r inferenceScreen

# Enable scrolling:
# Ctrl+A then Esc

# Kill all screens:
# pkill screen

### Docker Stuff ###

# Build image "image implementation_docker-web:latest":
# docker-compose build

# Create container "text_san" and run it from "image implementation_docker-web:latest" with SSL:
# docker run --name text_san -p 80:80 -p 443:443 -p 3050:3050 -v /etc/letsencrypt:/etc/letsencrypt -e SERVER_NAME=textsan.dimitristaufer.com -e CERT_PATH=/etc/letsencrypt/live/textsan.dimitristaufer.com/fullchain.pem -e KEY_PATH=/etc/letsencrypt/live/textsan.dimitristaufer.com/privkey.pem implementation_docker-web:latest
# Create container "text_san" and run it from "image implementation_docker-web:latest" without SSL:
# docker run --name text_san -p 80:80 -p 443:443 -p 3050:3050 implementation_docker-web:latest

# Stop the container:
# docker stop text_san

# Start the container:
# docker start text_san

# Attach to the container:
# docker attach text_san

# Delete the container:
# docker rm text_san

# Remove all Docker containers:
# docker rm -f $(docker ps -aq)

# Remove all Docker images:
# docker rmi -f $(docker images -aq)

# Remove the entire Docker cache:
# docker builder prune --all --force

# Show free space
# df -hT /dev/xvda1

### SSL Stuff ###

# sudo dnf install python3 augeas-libs
# sudo python3 -m venv /opt/certbot/
# sudo /opt/certbot/bin/pip install --upgrade pip
# sudo /opt/certbot/bin/pip install certbot certbot-nginx
# sudo ln -s /opt/certbot/bin/certbot /usr/bin/certbot
# sudo certbot certonly --standalone -d textsan.dimitristaufer.com (Note: Make sure nothing is running on port 80 when you run this command.)
# sudo certbot renew --dry-run
# sudo chmod -R 755 /etc/letsencrypt/live/
# sudo chmod -R 755 /etc/letsencrypt/archive/


# Base image with Python installed (slim version to reduce file size)
FROM python:3.9-slim

# Set default ENV variables for the SSL configuration
ENV SERVER_NAME=default
ENV CERT_PATH=/etc/letsencrypt/live/default/fullchain.pem
ENV KEY_PATH=/etc/letsencrypt/live/default/privkey.pem

# Set the working directory for the flask app
WORKDIR /app

# Update and install required packages
RUN apt-get update -y && apt-get install -y \
    python3-dev \
    nginx \
    gcc \
    g++ \
    build-essential \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    tk-dev \
    gettext \
    && rm -rf /var/lib/apt/lists/*
# Clean up to reduce image size

RUN pip install --upgrade setuptools

# Copy the application directory to the /app directory
COPY ./app /app

# Install any needed packages specified in requirements.txt
RUN CFLAGS="-march=x86-64" pip install --no-cache-dir -r /app/Backend/requirements.txt

# Copy the non-ssl nginx configuration
COPY ./app/Frontend/non-ssl.conf /etc/nginx/conf.d/

# Make port 80 and 3050 available outside this container
EXPOSE 80 3050

# Copy the start script to the docker container and make it executable
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Run start.sh when the container launches
CMD ["/start.sh"]
