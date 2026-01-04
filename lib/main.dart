import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'customers_page.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();

  final front = cameras.firstWhere(
    (c) => c.lensDirection == CameraLensDirection.front,
    orElse: () => cameras.first,
  );

  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.orange),
      home: HomePage(camera: front),
    ),
  );
}

class HomePage extends StatelessWidget {
  final CameraDescription camera;
  const HomePage({super.key, required this.camera});

  static const String baseUrl = "http://10.0.2.2:8000";

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Aplikacja lojalnościowa"),
        actions: [
          IconButton(
            icon: const Icon(Icons.people),
            tooltip: "Lista klientów",
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (_) => const CustomersPage(baseUrl: baseUrl),
                ),
              );
            },
          ),
        ],
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: SizedBox(
            width: 420,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.restaurant, size: 56),
                const SizedBox(height: 16),
                const Text(
                  "Witaj!",
                  style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                const Text(
                  "Kliknij, aby zeskanować twarz klienta i zarejestrować wizytę.",
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 24),
                ElevatedButton.icon(
                  icon: const Icon(Icons.face),
                  label: const Padding(
                    padding: EdgeInsets.symmetric(vertical: 14.0),
                    child: Text("SKANUJ TWARZ", style: TextStyle(fontSize: 18)),
                  ),
                  style: ElevatedButton.styleFrom(
                    minimumSize: const Size.fromHeight(56),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) =>
                            ScanPage(camera: camera, baseUrl: baseUrl),
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class ScanPage extends StatefulWidget {
  final CameraDescription camera;
  final String baseUrl;
  const ScanPage({super.key, required this.camera, required this.baseUrl});

  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> {
  static const int inputSize = 112;

  late CameraController _controller;

  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      performanceMode: FaceDetectorMode.accurate,
      enableLandmarks: false,
      enableClassification: false,
    ),
  );

  Interpreter? _interpreter;
  int? _embeddingDim;

  bool _isProcessing = false;

  String _status = "Kliknij „Skanuj”, aby rozpocząć";
  ScanResultView? _resultView;

  List<double>? _lastEmbedding;
  bool _notFound = false;

  @override
  void initState() {
    super.initState();
    _initCameraAndModel();
  }

  Future<void> _initCameraAndModel() async {
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
    );

    try {
      await _controller.initialize();

      _interpreter =
          await Interpreter.fromAsset('assets/models/mobilefacenet.tflite');

      final outShape = _interpreter!.getOutputTensor(0).shape;
      _embeddingDim = outShape.isNotEmpty ? outShape.last : null;

      setState(() {
        _status = "Gotowe. Możesz skanować.";
      });
    } catch (e) {
      setState(() {
        _status = "Błąd inicjalizacji: $e";
      });
    }
  }

  void _resetForNextScan() {
    setState(() {
      _status = "Kliknij „Skanuj”, aby zeskanować kolejną osobę";
      _resultView = null;
      _lastEmbedding = null;
      _notFound = false;
      _isProcessing = false;
    });
  }

  Future<void> _scanButtonPressed() async {
    if (_isProcessing) return;

    if (_resultView != null) {
      _resetForNextScan();
      await Future.delayed(const Duration(milliseconds: 80));
    }

    await _captureAndRecognize();
  }

  Future<void> _captureAndRecognize() async {
    if (_isProcessing) return;
    if (!_controller.value.isInitialized) return;
    if (_interpreter == null || _embeddingDim == null) {
      setState(() => _status = "Model TFLite nie jest załadowany.");
      return;
    }

    setState(() {
      _isProcessing = true;
      _status = "Robię zdjęcie...";
      _resultView = null;
      _lastEmbedding = null;
      _notFound = false;
    });

    try {
      final XFile photo = await _controller.takePicture();

      final inputImage = InputImage.fromFilePath(photo.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        setState(() {
          _status = "Nie wykryto twarzy. Spróbuj ponownie.";
        });
        return;
      }

      faces.sort((a, b) => (b.boundingBox.width * b.boundingBox.height)
          .compareTo(a.boundingBox.width * a.boundingBox.height));
      final face = faces.first;

      setState(() => _status = "Rozpoznanie...");

      final embedding = await _computeEmbedding(photo.path, face.boundingBox);

      setState(() => _status = "Sprawdzanie w bazie...");
      await _sendEmbeddingToBackend(embedding);
    } catch (e) {
      setState(() => _status = "Błąd: $e");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<List<double>> _computeEmbedding(String photoPath, Rect faceBox) async {
    final bytes = await File(photoPath).readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw Exception("Nie udało się odczytać skanu.");
    }

    final cropped = _cropWithPadding(decoded, faceBox, paddingPercent: 0.25);
    final resized = img.copyResizeCropSquare(cropped, size: inputSize);

    final input = _imageToFloat32Normalized(resized);
    final emb = _runTflite(input);

    final norm = math.sqrt(emb.fold(0.0, (s, v) => s + v * v));
    if (norm == 0) return emb;
    return emb.map((v) => v / norm).toList(growable: false);
  }

  List<double> _runTflite(Float32List input) {
    final dim = _embeddingDim!;
    final inputTensor = input.reshape([1, inputSize, inputSize, 3]);
    final output = List.filled(dim, 0.0).reshape([1, dim]);
    _interpreter!.run(inputTensor, output);
    return List<double>.from(output[0] as List);
  }

  img.Image _cropWithPadding(img.Image src, Rect box,
      {double paddingPercent = 0.25}) {
    final padW = box.width * paddingPercent;
    final padH = box.height * paddingPercent;

    int x = (box.left - padW).round();
    int y = (box.top - padH).round();
    int w = (box.width + 2 * padW).round();
    int h = (box.height + 2 * padH).round();

    x = x.clamp(0, src.width - 1);
    y = y.clamp(0, src.height - 1);
    w = math.min(w, src.width - x);
    h = math.min(h, src.height - y);

    return img.copyCrop(src, x: x, y: y, width: w, height: h);
  }

  Float32List _imageToFloat32Normalized(img.Image image) {
    final Float32List input = Float32List(inputSize * inputSize * 3);
    int idx = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final p = image.getPixel(x, y);
        final r = p.r.toDouble();
        final g = p.g.toDouble();
        final b = p.b.toDouble();

        input[idx++] = (r - 127.5) / 127.5;
        input[idx++] = (g - 127.5) / 127.5;
        input[idx++] = (b - 127.5) / 127.5;
      }
    }
    return input;
  }

  String _formatDateTime(String? iso) {
    if (iso == null || iso.isEmpty) return "—";
    try {
      final dt = DateTime.parse(iso).toLocal();
      String two(int n) => n.toString().padLeft(2, "0");
      return "${two(dt.day)}.${two(dt.month)}.${dt.year}  ${two(dt.hour)}:${two(dt.minute)}";
    } catch (_) {
      return iso;
    }
  }

  Future<void> _sendEmbeddingToBackend(List<double> embedding) async {
    try {
      final uri = Uri.parse("${widget.baseUrl}/scan-visit");
      final resp = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "embedding": embedding,
          "device_id": "android_emulator",
        }),
      );

      if (resp.statusCode != 200) {
        setState(() {
          _status = "Serwer odrzucił żądanie (${resp.statusCode}).";
          _resultView = ScanResultView.error(resp.body);
        });
        return;
      }

      final data = jsonDecode(resp.body);

      if (data["status"] == "not_found") {
        setState(() {
          _status = "Nie znaleziono klienta.";
          _notFound = true;
          _lastEmbedding = embedding;
          _resultView = ScanResultView.notFound();
        });
        return;
      }

      final displayName = (data["display_name"] ?? "—").toString();
      final stateLabel = (data["state_label"] ?? "stały klient").toString();
      final sinceReward = (data["visits_since_reward"] ?? 0) as int;

      final remaining = (5 - sinceReward);
      final lastVisitAt =
          _formatDateTime((data["last_visit_at"] ?? "").toString());

      final reward = data["reward"] == true;

      setState(() {
        _status = "Zeskanowano.";
        _notFound = false;
        _lastEmbedding = null;

        _resultView = ScanResultView.matched(
          stateLabel: _capitalizeFirst(stateLabel),
          name: displayName,
          visitsSinceReward: sinceReward,
          remainingToCoffee: remaining <= 0 ? 0 : remaining,
          lastVisitAt: lastVisitAt,
        );
      });

      if (reward) {
        await _showFreeCoffeeDialog();
      }
    } catch (_) {
      setState(() {
        _status = "Brak połączenia z serwerem.";
        _resultView =
            ScanResultView.error("Nie można połączyć się z backendem.");
      });
    }
  }

  Future<void> _showFreeCoffeeDialog() async {
    if (!mounted) return;

    await showGeneralDialog(
      context: context,
      barrierDismissible: true,
      barrierLabel: "DARMOWA KAWA",
      barrierColor: Colors.black54,
      transitionDuration: const Duration(milliseconds: 220),
      pageBuilder: (_, __, ___) {
        return Center(
          child: Material(
            color: Colors.transparent,
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 24),
              padding: const EdgeInsets.all(22),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(24),
                boxShadow: const [
                  BoxShadow(
                    blurRadius: 18,
                    spreadRadius: 2,
                    offset: Offset(0, 8),
                    color: Colors.black26,
                  ),
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.celebration, size: 54, color: Colors.orange),
                  const SizedBox(height: 10),
                  const Text(
                    "GRATULACJE!",
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.w900),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 10),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 14),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFFF3E0),
                      borderRadius: BorderRadius.circular(18),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text("☕️", style: TextStyle(fontSize: 34)),
                        SizedBox(width: 10),
                        Text(
                          "DARMOWA KAWA",
                          style: TextStyle(
                              fontSize: 20, fontWeight: FontWeight.w800),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    "Klient uzbierał 5 wizyt.\nWydaj nagrodę przy kasie 🙂",
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 16),
                  ),
                  const SizedBox(height: 18),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                      ),
                      onPressed: () => Navigator.of(context).pop(),
                      child: const Text("OK", style: TextStyle(fontSize: 16)),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
      transitionBuilder: (_, animation, __, child) {
        final curved =
            CurvedAnimation(parent: animation, curve: Curves.easeOutBack);
        return FadeTransition(
          opacity: animation,
          child: ScaleTransition(scale: curved, child: child),
        );
      },
    );
  }

  String _capitalizeFirst(String s) {
    if (s.isEmpty) return s;
    return s[0].toUpperCase() + s.substring(1);
  }

  Future<String?> _askForName() async {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Imię klienta"),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(hintText: "np. Jan"),
          textInputAction: TextInputAction.done,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, null),
            child: const Text("Anuluj"),
          ),
          ElevatedButton(
            onPressed: () {
              final name = controller.text.trim();
              Navigator.pop(ctx, name.isEmpty ? null : name);
            },
            child: const Text("Zapisz"),
          ),
        ],
      ),
    );
  }

  Future<void> _enrollNewCustomer() async {
    final emb = _lastEmbedding;
    if (emb == null) return;

    final name = await _askForName();
    if (name == null) return;

    setState(() {
      _isProcessing = true;
      _status = "Rejestrowanie nowego klienta...";
    });

    try {
      final uri = Uri.parse("${widget.baseUrl}/enroll");
      final resp = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "embeddings": [emb],
          "display_name": name,
        }),
      );

      if (resp.statusCode != 200) {
        setState(() {
          _status = "Błąd enroll (${resp.statusCode})";
          _resultView = ScanResultView.error(resp.body);
        });
        return;
      }

      setState(() {
        _status = "Utworzono klienta: $name ✅";
        _resultView = ScanResultView.info(
          "Zarejestrowano klienta.\nKliknij „Skanuj”, aby nabić pierwszą wizytę.",
        );
        _lastEmbedding = null;
        _notFound = false;
      });
    } catch (_) {
      setState(() {
        _status = "Brak połączenia z serwerem.";
        _resultView =
            ScanResultView.error("Nie można połączyć się z backendem.");
      });
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    _faceDetector.close();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized && !_status.startsWith("Błąd")) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final buttonText = _resultView == null ? "SKANUJ" : "SKANUJ NASTĘPNEGO";

    return Scaffold(
      appBar: AppBar(
        title: const Text("Skanowanie"),
        actions: [
          IconButton(
            icon: const Icon(Icons.people),
            tooltip: "Lista klientów",
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (_) => CustomersPage(baseUrl: widget.baseUrl),
                ),
              );
            },
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(flex: 3, child: CameraPreview(_controller)),
          Expanded(
            flex: 2,
            child: Container(
              padding: const EdgeInsets.all(18),
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.vertical(top: Radius.circular(26)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    _status,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: Colors.orange,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: _resultView == null
                        ? const Center(
                            child: Text(
                              "Zeskanuj klienta, aby zobaczyć wynik.",
                              textAlign: TextAlign.center,
                            ),
                          )
                        : SingleChildScrollView(
                            padding: const EdgeInsets.only(bottom: 8),
                            child: _resultView!.buildWidget(context),
                          ),
                  ),
                  const SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: _isProcessing ? null : _scanButtonPressed,
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                    ),
                    child: _isProcessing
                        ? const SizedBox(
                            height: 18,
                            width: 18,
                            child: CircularProgressIndicator(
                              color: Colors.white,
                              strokeWidth: 2,
                            ),
                          )
                        : Text(buttonText,
                            style: const TextStyle(fontSize: 16)),
                  ),
                  const SizedBox(height: 8),
                  if (_notFound && _lastEmbedding != null)
                    OutlinedButton.icon(
                      onPressed: _isProcessing ? null : _enrollNewCustomer,
                      icon: const Icon(Icons.person_add),
                      label: const Text("ZAREJESTRUJ NOWEGO KLIENTA"),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
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

class ScanResultView {
  final Widget Function(BuildContext) _builder;
  ScanResultView._(this._builder);

  Widget buildWidget(BuildContext context) => _builder(context);

  factory ScanResultView.notFound() {
    return ScanResultView._((_) => Card(
          elevation: 0,
          color: const Color(0xFFFFF3E0),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: const Padding(
            padding: EdgeInsets.all(16),
            child: Text(
              "Nie znaleziono klienta w bazie.\nMożesz go zarejestrować.",
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
          ),
        ));
  }

  factory ScanResultView.info(String msg) {
    return ScanResultView._((_) => Card(
          elevation: 0,
          color: const Color(0xFFE3F2FD),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
              msg,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 16),
            ),
          ),
        ));
  }

  factory ScanResultView.error(String msg) {
    return ScanResultView._((_) => Card(
          elevation: 0,
          color: const Color(0xFFFFEBEE),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Text(msg, style: const TextStyle(fontSize: 14)),
          ),
        ));
  }

  factory ScanResultView.matched({
    required String stateLabel,
    required String name,
    required int visitsSinceReward,
    required int remainingToCoffee,
    required String lastVisitAt,
  }) {
    return ScanResultView._((_) => Card(
          elevation: 0,
          color: const Color(0xFFF1F8E9),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  stateLabel,
                  style: const TextStyle(
                      fontSize: 16, fontWeight: FontWeight.w800),
                ),
                const SizedBox(height: 10),
                Text("Imię: $name", style: const TextStyle(fontSize: 16)),
                const SizedBox(height: 6),
                Text("Ilość wizyt: $visitsSinceReward/5",
                    style: const TextStyle(fontSize: 16)),
                const SizedBox(height: 6),
                Text("Brakuje do darmowej kawy: $remainingToCoffee",
                    style: const TextStyle(fontSize: 16)),
                const SizedBox(height: 6),
                Text("Czas ostatniej wizyty: $lastVisitAt",
                    style: const TextStyle(fontSize: 16)),
              ],
            ),
          ),
        ));
  }
}
