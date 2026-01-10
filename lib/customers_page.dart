import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class AppColors {
  static const brownDark = Color(0xFF3E2723);
  static const creamBg = Color(0xFFFFF8E1);
  static const orangeAccent = Color(0xFFEF6C00);
}

class CustomersPage extends StatefulWidget {
  final String baseUrl;
  const CustomersPage({super.key, required this.baseUrl});

  @override
  State<CustomersPage> createState() => _CustomersPageState();
}

class _CustomersPageState extends State<CustomersPage> {
  List<dynamic> _customers = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetch();
  }

  Future<void> _fetch() async {
    setState(() => _isLoading = true);
    try {
      final resp = await http.get(Uri.parse("${widget.baseUrl}/customers"));
      if (resp.statusCode == 200) {
        setState(() => _customers = jsonDecode(resp.body));
      }
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _delete(String id, String name) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Usuwanie klienta"),
        content: Text("Czy na pewno usunąć $name z bazy?"),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text("Anuluj")),
          TextButton(onPressed: () => Navigator.pop(ctx, true), child: const Text("Usuń", style: TextStyle(color: Colors.red))),
        ],
      ),
    );

    if (confirm == true) {
      await http.delete(Uri.parse("${widget.baseUrl}/customers/$id"));
      _fetch();
    }
  }

  String _formatDate(String? iso) {
    if (iso == null || iso.isEmpty) return "Brak wizyt";
    try {
      final dt = DateTime.parse(iso).toLocal();
      return "${dt.day}.${dt.month}.${dt.year} ${dt.hour}:${dt.minute.toString().padLeft(2, '0')}";
    } catch (_) { return "—"; }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.creamBg,
      appBar: AppBar(title: const Text("Baza Klientów"), centerTitle: true, backgroundColor: Colors.transparent),
      body: _isLoading 
        ? const Center(child: CircularProgressIndicator())
        : RefreshIndicator(
            onRefresh: _fetch,
            child: ListView.builder(
              padding: const EdgeInsets.all(15),
              itemCount: _customers.length,
              itemBuilder: (context, index) {
                final c = _customers[index];
                return Card(
                  margin: const EdgeInsets.only(bottom: 15),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            const CircleAvatar(backgroundColor: AppColors.brownDark, child: Icon(Icons.person, color: Colors.white)),
                            const SizedBox(width: 15),
                            Expanded(child: Text(c["display_name"] ?? "Anonim", style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold))),
                            IconButton(icon: const Icon(Icons.delete_outline, color: Colors.red), onPressed: () => _delete(c["id"], c["display_name"]))
                          ],
                        ),
                        const Divider(height: 25),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            _statColumn("Darmowa kawa", "${c["visits_since_reward"]}/5"),
                            _statColumn("Wszystkie wizyty", "${c["visits_total"]}"),
                            _statColumn("Ostatnia wizyta", _formatDate(c["last_visit_at"])),
                          ],
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
    );
  }

  Widget _statColumn(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontSize: 11, color: Colors.grey, fontWeight: FontWeight.bold)),
        Text(value, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: AppColors.brownDark)),
      ],
    );
  }
}