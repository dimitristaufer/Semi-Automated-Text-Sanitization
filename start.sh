#!/bin/bash

# Define the locations of the certificate and key
CERT=$CERT_PATH
KEY=$KEY_PATH

# Remove any default configurations if they exist
rm -f /etc/nginx/sites-enabled/default

# Replace the ssl variables to ssl.template and write its content to ssl.conf
# This allows us to specify the certificates through the "docker run" command
IFS='$' envsubst '${SERVER_NAME},${CERT_PATH},${KEY_PATH}' < /app/Frontend/ssl.template > /app/Frontend/ssl.conf

# Print the generated nginx configuration for debugging
#cat /app/Frontend/ssl.conf

# Check if the certificate and key files exist
if [ -f "$CERT" ] && [ -f "$KEY" ]; then
    # If they exist, use the SSL configuration
    echo "SSL certificate and key found. Configuring Nginx for SSL..."
    cp /app/Frontend/ssl.conf /etc/nginx/conf.d/default.conf
    rm -f /etc/nginx/conf.d/non-ssl.conf
else
    # If they don't exist, use the non-SSL configuration
    echo "SSL certificate and key not found. Configuring Nginx without SSL..."
    cp /etc/nginx/conf.d/non-ssl.conf /etc/nginx/conf.d/default.conf
    rm -f /etc/nginx/conf.d/ssl.conf
fi

# Start Nginx in the background
echo "Starting Nginx..."
nginx -g "daemon off;" &

# Start Flask app in the foreground
echo "Starting Flask app..."
exec python /app/Backend/server.py