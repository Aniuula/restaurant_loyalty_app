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
import 'package:audioplayers/audioplayers.dart';

import 'customers_page.dart';

class AppColors {
  static const brownDark = Color(0xFF3E2723);
  static const brownMedium = Color(0xFF5D4037);
  static const orangeAccent = Color(0xFFEF6C00);
  static const creamBg = Color(0xFFFFF8E1);
  static const gold = Color(0xFFFFD700);
}

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
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: AppColors.creamBg,
        colorScheme: ColorScheme.fromSeed(seedColor: AppColors.brownDark),
      ),
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
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(30),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 40),
              const Text("Loyalty\nCoffee", 
                style: TextStyle(fontSize: 42, fontWeight: FontWeight.w900, height: 1.1, color: AppColors.brownDark)),
              const SizedBox(height: 10),
              const Text("Program lojalnościowy", 
                style: TextStyle(fontSize: 16, color: AppColors.brownMedium)),
              const Spacer(),
              _buildMenuCard(context, "SKANOWANIE", Icons.face_unlock_rounded, AppColors.brownDark, 
                () => Navigator.push(context, MaterialPageRoute(builder: (_) => ScanPage(camera: camera, baseUrl: baseUrl)))),
              const SizedBox(height: 20),
              _buildMenuCard(context, "BAZA KLIENTÓW", Icons.people_alt_rounded, AppColors.brownMedium, 
                () => Navigator.push(context, MaterialPageRoute(builder: (_) => CustomersPage(baseUrl: baseUrl)))),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMenuCard(BuildContext context, String title, IconData icon, Color color, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(25),
      child: Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: Colors.white, 
          borderRadius: BorderRadius.circular(25), 
          boxShadow: [BoxShadow(color: color.withOpacity(0.1), blurRadius: 20, offset: const Offset(0, 10))]
        ),
        child: Row(children: [
          Icon(icon, color: color, size: 30), 
          const SizedBox(width: 20), 
          Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18, color: AppColors.brownDark))
        ]),
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
  late CameraController _controller;
  final FaceDetector _faceDetector = FaceDetector(options: FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate));
  final AudioPlayer _audioPlayer = AudioPlayer();
  Interpreter? _interpreter;

  bool _isProcessing = false;
  String _status = "Ustaw twarz w kadrze";
  Widget? _resultCard;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    _controller = CameraController(widget.camera, ResolutionPreset.high, enableAudio: false);
    await _controller.initialize();
    _interpreter = await Interpreter.fromAsset('assets/models/mobilefacenet.tflite');
    if (mounted) setState(() {});
  }

  void _resetScanner() {
    setState(() {
      _resultCard = null;
      _status = "Ustaw twarz w kadrze";
      _isProcessing = false;
    });
  }

  Future<void> _manualScan() async {
    if (_isProcessing) return;
    setState(() { _isProcessing = true; _status = "Analizowanie..."; _resultCard = null; });

    try {
      final photo = await _controller.takePicture();
      final inputImage = InputImage.fromFilePath(photo.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        setState(() { _status = "Nie wykryto twarzy!"; _isProcessing = false; });
        return;
      }

      final face = faces.first;
      final bytes = await File(photo.path).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) return;

      final cropped = img.copyCrop(decoded, 
          x: face.boundingBox.left.toInt(), 
          y: face.boundingBox.top.toInt(), 
          width: face.boundingBox.width.toInt(), 
          height: face.boundingBox.height.toInt());
      final resized = img.copyResize(cropped, width: 112, height: 112);
      
      final input = _imageToFloat32(resized);
      final output = List.filled(192, 0.0).reshape([1, 192]);
      _interpreter!.run(input.reshape([1, 112, 112, 3]), output);

      List<double> emb = List<double>.from(output[0]);
      double norm = math.sqrt(emb.fold(0, (p, c) => p + c * c));
      final normalizedEmb = emb.map((e) => e / norm).toList();

      final resp = await http.post(Uri.parse("${widget.baseUrl}/scan-visit"), 
          headers: {"Content-Type": "application/json"}, 
          body: jsonEncode({"embedding": normalizedEmb}));
      final data = jsonDecode(resp.body);

      if (data["status"] == "matched") {
        _audioPlayer.play(AssetSource('sounds/success.mp3'));
        setState(() { 
          _status = "WITAJ PONOWNIE!"; 
          _resultCard = _buildMatchedCard(data["display_name"], data["visits_since_reward"]); 
        });
        if (data["reward"] == true) _showCelebrationDialog();
      } else {
        setState(() { 
          _status = "NOWY GOŚĆ"; 
          _resultCard = _buildNewGuestCard(normalizedEmb); 
        });
      }
    } catch (e) { setState(() => _status = "BŁĄD POŁĄCZENIA"); }
    finally { setState(() => _isProcessing = false); }
  }

  Float32List _imageToFloat32(img.Image image) {
    var buffer = Float32List(112 * 112 * 3);
    int idx = 0;
    for (var y = 0; y < 112; y++) {
      for (var x = 0; x < 112; x++) {
        var p = image.getPixel(x, y);
        buffer[idx++] = (p.r - 127.5) / 127.5;
        buffer[idx++] = (p.g - 127.5) / 127.5;
        buffer[idx++] = (p.b - 127.5) / 127.5;
      }
    }
    return buffer;
  }

  Widget _buildMatchedCard(String name, int visits) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white, 
        borderRadius: BorderRadius.circular(25), 
        boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 15)]
      ),
      child: Column(children: [
        Text(name, style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: AppColors.brownDark)),
        const SizedBox(height: 15),
        Row(
          mainAxisAlignment: MainAxisAlignment.center, 
          children: List.generate(5, (i) => Icon(
            Icons.local_cafe, 
            color: i < visits ? AppColors.orangeAccent : Colors.grey[300], 
            size: 35
          ))
        ),
        const SizedBox(height: 10),
        Text("Wizyty: $visits/5", style: const TextStyle(color: AppColors.brownMedium, fontWeight: FontWeight.bold)),
      ]),
    );
  }

  Widget _buildNewGuestCard(List<double> emb) {
    return Column(
      children: [
        const Text("Brak klienta w bazie", 
            style: TextStyle(color: AppColors.brownMedium, fontWeight: FontWeight.w500)),
        const SizedBox(height: 15),
        _buildActionButton(
          title: "REJESTRACJA",
          icon: Icons.person_add_alt_1,
          color: AppColors.orangeAccent,
          onPressed: () => _enrollAndVisit(emb),
        ),
      ],
    );
  }

  Future<void> _enrollAndVisit(List<double> emb) async {
    final ctrl = TextEditingController();
    bool consent = false;
    await showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => StatefulBuilder(
        builder: (context, setDS) => AlertDialog(
          backgroundColor: AppColors.creamBg,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(25)),
          title: const Text("Nowa Rejestracja", style: TextStyle(fontWeight: FontWeight.bold)),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(controller: ctrl, decoration: const InputDecoration(labelText: "Imię klienta")),
              const SizedBox(height: 20),
              Row(children: [
                Checkbox(value: consent, onChanged: (v) => setDS(() => consent = v!)),
                const Expanded(child: Text("Zgoda na skan twarzy (RODO).", style: TextStyle(fontSize: 12))),
              ]),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx), child: const Text("Anuluj")),
            ElevatedButton(
              onPressed: !consent ? null : () async {
                if (ctrl.text.isEmpty) return;
                await http.post(Uri.parse("${widget.baseUrl}/enroll"), 
                    headers: {"Content-Type": "application/json"}, 
                    body: jsonEncode({"embeddings": [emb], "display_name": ctrl.text}));
                final resp = await http.post(Uri.parse("${widget.baseUrl}/scan-visit"), 
                    headers: {"Content-Type": "application/json"}, 
                    body: jsonEncode({"embedding": emb}));
                final data = jsonDecode(resp.body);
                
                Navigator.pop(ctx);
                _audioPlayer.play(AssetSource('sounds/success.mp3'));
                setState(() {
                  _status = "WITAMY W PROGRAMIE!";
                  _resultCard = _buildMatchedCard(ctrl.text, data["visits_since_reward"]);
                });
              },
              child: const Text("Zapisz"),
            ),
          ],
        ),
      ),
    );
  }

  void _showCelebrationDialog() {
    showGeneralDialog(
      context: context,
      barrierDismissible: true,
      barrierLabel: "Nagroda",
      transitionDuration: const Duration(milliseconds: 400),
      pageBuilder: (context, anim1, anim2) {
        return Center(
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 30),
            padding: const EdgeInsets.all(25),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(30),
              boxShadow: [BoxShadow(color: AppColors.gold.withOpacity(0.5), blurRadius: 20, spreadRadius: 5)]
            ),
            child: Material(
              color: Colors.transparent,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text("🎉", style: TextStyle(fontSize: 60)),
                  const SizedBox(height: 10),
                  const Text("DARMOWA KAWA!", 
                      style: TextStyle(fontSize: 26, fontWeight: FontWeight.w900, color: AppColors.orangeAccent)),
                  const SizedBox(height: 10),
                  const Text("Klient uzbierał komplet 5 wizyt", 
                      textAlign: TextAlign.center, style: TextStyle(fontSize: 16)),
                  const SizedBox(height: 25),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: () => Navigator.pop(context),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.brownDark,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 15),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15))
                      ),
                      child: const Text("WYDAJ NAGRODĘ", style: TextStyle(fontWeight: FontWeight.bold)),
                    ),
                  )
                ],
              ),
            ),
          ),
        );
      },
      transitionBuilder: (context, anim1, anim2, child) {
        return ScaleTransition(
          scale: CurvedAnimation(parent: anim1, curve: Curves.elasticOut),
          child: FadeTransition(opacity: anim1, child: child),
        );
      },
    );
  }

  Widget _buildActionButton({required String title, required IconData icon, required Color color, required VoidCallback? onPressed}) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20), 
        color: color, 
        boxShadow: [BoxShadow(color: color.withOpacity(0.3), blurRadius: 10, offset: const Offset(0, 5))]
      ),
      child: ElevatedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, color: Colors.white),
        label: Text(title, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.transparent, 
          shadowColor: Colors.transparent, 
          foregroundColor: Colors.white, 
          minimumSize: const Size(double.infinity, 60)
        ),
      ),
    );
  }

  @override
  void dispose() { _controller.dispose(); _faceDetector.close(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          Positioned.fill(
            bottom: MediaQuery.of(context).size.height * 0.4,
            child: ClipRRect(
              borderRadius: const BorderRadius.vertical(bottom: Radius.circular(40)),
              child: FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: _controller.value.previewSize!.height,
                  height: _controller.value.previewSize!.width,
                  child: Transform.scale(
                    scaleX: -1.0, 
                    child: CameraPreview(_controller),
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            top: 50, left: 20, 
            child: GestureDetector(
              onTap: () => Navigator.pop(context), 
              child: Container(
                padding: const EdgeInsets.all(10), 
                decoration: BoxDecoration(color: Colors.black38, borderRadius: BorderRadius.circular(15)), 
                child: const Icon(Icons.home_rounded, color: Colors.white)
              )
            )
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              height: MediaQuery.of(context).size.height * 0.45,
              width: double.infinity,
              padding: const EdgeInsets.all(30),
              decoration: const BoxDecoration(
                color: AppColors.creamBg, 
                borderRadius: BorderRadius.vertical(top: Radius.circular(40))
              ),
              child: Column(
                children: [
                  Text(_status, textAlign: TextAlign.center, 
                      style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.brownDark)),
                  const SizedBox(height: 20),
                  if (_resultCard != null) _resultCard!,
                  const Spacer(),
                  if (_resultCard == null)
                    _buildActionButton(
                      title: "SKANUJ TWARZ", 
                      icon: Icons.camera_alt, 
                      color: AppColors.brownDark, 
                      onPressed: _isProcessing ? null : _manualScan
                    )
                  else
                    _buildActionButton(
                      title: "SKANOWANIE KLIENTA", 
                      icon: Icons.refresh_rounded, 
                      color: AppColors.orangeAccent, 
                      onPressed: _resetScanner
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