import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:http/http.dart' as http;

void main() async {
  // Inicjalizacja silnika Flutter i kamer
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();

  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.orange),
      home: RestaurantFaceScanner(
        camera: cameras[1],
      ), // Używamy przedniej kamery
    ),
  );
}

class RestaurantFaceScanner extends StatefulWidget {
  final CameraDescription camera;
  const RestaurantFaceScanner({super.key, required this.camera});

  @override
  State<RestaurantFaceScanner> createState() => _RestaurantFaceScannerState();
}

class _RestaurantFaceScannerState extends State<RestaurantFaceScanner> {
  late CameraController _controller;
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(),
  );
  bool _isProcessing = false;
  String _statusMessage = "Ustaw twarz i kliknij przycisk";
  String _customerInfo = "";

  @override
  void initState() {
    super.initState();
    _controller = CameraController(widget.camera, ResolutionPreset.high);
    _controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
    });
  }

  // Funkcja wykonująca zdjęcie i wysyłająca je do serwera
  Future<void> _captureAndRecognize() async {
    if (_isProcessing) return;

    setState(() {
      _isProcessing = true;
      _statusMessage = "Przetwarzanie obrazu...";
    });

    try {
      // 1. Zrób zdjęcie
      final XFile photo = await _controller.takePicture();

      // 2. Lokalna weryfikacja: Czy na zdjęciu jest twarz?
      final inputImage = InputImage.fromFilePath(photo.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        setState(() {
          _statusMessage = "BŁĄD: Nie wykryto twarzy!";
          _customerInfo = "";
        });
      } else {
        setState(() => _statusMessage = "Twarz wykryta. Rozpoznaję...");
        // 3. Wyślij zdjęcie do serwera FastAPI
        await _uploadToBackend(photo.path);
      }
    } catch (e) {
      setState(() => _statusMessage = "Błąd techniczny: $e");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  // Komunikacja z serwerem Python
  Future<void> _uploadToBackend(String path) async {
    try {
      // Adres 10.0.2.2 jest specjalnym adresem komputera dla emulatora Android
      var uri = Uri.parse('http://10.0.2.2:8000/scan');
      var request = http.MultipartRequest('POST', uri);

      // Dodajemy plik zdjęcia do paczki
      request.files.add(await http.MultipartFile.fromPath('file', path));

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);

        setState(() {
          _statusMessage = "SUKCES!";
          _customerInfo =
              "Status: ${data['status']}\n"
              "ID: ${data['id']}\n"
              "Punkty: ${data['punkty']}/5\n"
              "${data['nagroda'] ? '🎁 DARMOWA KAWA!' : ''}";
        });
      } else {
        setState(() => _statusMessage = "Serwer odrzucił żądanie.");
      }
    } catch (e) {
      setState(() => _statusMessage = "Brak połączenia z serwerem Python.");
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text("System Lojalnościowy AI")),
      body: Column(
        children: [
          // Podgląd z kamery
          Expanded(flex: 3, child: CameraPreview(_controller)),
          // Panel informacyjny i przycisk
          Expanded(
            flex: 2,
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.vertical(top: Radius.circular(30)),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    _statusMessage,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.orange,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    _customerInfo,
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 16, color: Colors.black87),
                  ),
                  const Spacer(),
                  ElevatedButton(
                    onPressed: _isProcessing ? null : _captureAndRecognize,
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 50,
                        vertical: 20,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(15),
                      ),
                    ),
                    child: _isProcessing
                        ? const CircularProgressIndicator(color: Colors.white)
                        : const Text(
                            "SKANUJ KLIENTA",
                            style: TextStyle(fontSize: 18),
                          ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
