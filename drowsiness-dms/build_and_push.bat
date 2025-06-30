@echo off
echo Building and pushing Docker image for Drowsiness Detection API

echo.
echo Step 1: Building Docker image...
docker build -t sshubham2004/okdriver:drowsiness-dms .

echo.
echo Step 2: Logging in to Docker Hub...
echo Please enter your Docker Hub credentials when prompted
docker login

echo.
echo Step 3: Pushing image to Docker Hub...
docker push sshubham2004/okdriver:drowsiness-dms

echo.
echo Process completed!
echo The image is now available at: sshubham2004/okdriver:drowsiness-dms

pause