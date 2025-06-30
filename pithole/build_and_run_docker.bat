@echo off
echo Building Docker image for Pothole API...

:: Build the Docker image
docker build -t pothole-api .

echo.
echo Docker image built successfully!
echo.

echo Running Pothole API container...
:: Run the container, mapping port 8000 to host port 8000
docker run -p 8000:8000 --name pothole-api-container pothole-api

:: Note: Press Ctrl+C to stop the container