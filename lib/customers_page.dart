import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class CustomersPage extends StatefulWidget {
  final String baseUrl; // np. http://10.0.2.2:8000
  const CustomersPage({super.key, required this.baseUrl});

  @override
  State<CustomersPage> createState() => _CustomersPageState();
}

class _CustomersPageState extends State<CustomersPage> {
  bool _loading = true;
  String? _error;
  List<dynamic> _customers = const [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final uri = Uri.parse("${widget.baseUrl}/customers");
      final resp = await http.get(uri);

      if (resp.statusCode != 200) {
        setState(() {
          _error = "Błąd serwera (${resp.statusCode})";
          _loading = false;
        });
        return;
      }

      final data = jsonDecode(resp.body);
      if (data is! List) {
        setState(() {
          _error = "Niepoprawny format odpowiedzi (oczekiwano listy).";
          _loading = false;
        });
        return;
      }

      setState(() {
        _customers = data;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = "Brak połączenia / błąd: $e";
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Klienci"),
        actions: [
          IconButton(
            onPressed: _loading ? null : _load,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text(
                      _error!,
                      textAlign: TextAlign.center,
                      style: const TextStyle(fontSize: 16),
                    ),
                  ),
                )
              : _customers.isEmpty
                  ? const Center(child: Text("Brak klientów w bazie."))
                  : ListView.separated(
                      itemCount: _customers.length,
                      separatorBuilder: (_, __) => const Divider(height: 1),
                      itemBuilder: (context, i) {
                        final c = _customers[i] as Map<String, dynamic>;
                        final id = (c["id"] ?? "").toString();
                        final name = (c["display_name"] ?? "—").toString();
                        final visitsTotal = c["visits_total"] ?? 0;
                        final visitsSince = c["visits_since_reward"] ?? 0;

                        final createdAt = (c["created_at"] ?? "").toString();
                        final updatedAt = (c["updated_at"] ?? "").toString();

                        return ListTile(
                          leading: const Icon(Icons.person),
                          title: Text(name),
                          subtitle: Text(
                            "ID: $id\n"
                            "Wizyty: $visitsTotal | Do nagrody: $visitsSince/5\n"
                            "Utw.: $createdAt\nAkt.: $updatedAt",
                          ),
                          isThreeLine: true,
                          onTap: () {
                            showDialog(
                              context: context,
                              builder: (_) => AlertDialog(
                                title: const Text("Klient"),
                                content: SelectableText(
                                  const JsonEncoder.withIndent("  ").convert(c),
                                ),
                                actions: [
                                  TextButton(
                                    onPressed: () => Navigator.pop(context),
                                    child: const Text("Zamknij"),
                                  ),
                                ],
                              ),
                            );
                          },
                        );
                      },
                    ),
    );
  }
}
