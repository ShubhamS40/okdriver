# Pothole API Docker Container

इस फोल्डर में Pothole API को Docker container में चलाने के लिए आवश्यक फाइलें हैं।

## आवश्यकताएँ

- Docker Desktop इंस्टॉल होना चाहिए
- Windows पर Docker Desktop चल रहा होना चाहिए

## Docker Container बनाने और चलाने के तरीके

### विधि 1: build_and_run_docker.bat का उपयोग करके

1. `build_and_run_docker.bat` फाइल पर डबल-क्लिक करें
2. यह स्क्रिप्ट Docker image बनाएगी और फिर container चलाएगी
3. Container को रोकने के लिए कमांड विंडो में Ctrl+C दबाएं

```
build_and_run_docker.bat
```

### विधि 2: Docker Compose का उपयोग करके

1. `run_with_compose.bat` फाइल पर डबल-क्लिक करें
2. यह स्क्रिप्ट Docker Compose का उपयोग करके container चलाएगी
3. Container को रोकने के लिए कमांड विंडो में Ctrl+C दबाएं

```
run_with_compose.bat
```

### मैन्युअल कमांड्स

यदि आप कमांड लाइन से चलाना चाहते हैं:

```bash
# Docker image बनाने के लिए
docker build -t pothole-api .

# Container चलाने के लिए
docker run -p 8000:8000 --name pothole-api-container pothole-api

# या Docker Compose के साथ
docker-compose up
```

## API एंडपॉइंट्स

Container चलने के बाद, निम्नलिखित URL पर API उपलब्ध होगी:

- API स्वास्थ्य जांच: http://localhost:8000/health
- API रूट: http://localhost:8000/
- Pothole डिटेक्शन एंडपॉइंट: http://localhost:8000/detect_pothole (POST request)

## मोबाइल ऐप कनेक्शन

मोबाइल ऐप को API से कनेक्ट करने के लिए, `pothole_service.dart` फाइल में API URL को अपडेट करें:

```dart
// API endpoints
final String _apiUrl = 'http://localhost:8000/detect_pothole';
final String _webSocketUrl = 'ws://localhost:8000/ws';
```

यदि आप एमुलेटर पर चला रहे हैं, तो `localhost` के बजाय `10.0.2.2` (Android) या `127.0.0.1` (iOS) का उपयोग करें।

यदि आप फिजिकल डिवाइस पर चला रहे हैं, तो अपने कंप्यूटर के IP एड्रेस का उपयोग करें।