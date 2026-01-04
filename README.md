# Restaurant Loyalty App (Face Scan)

Flutter + FastAPI demo of a restaurant loyalty system:
- Flutter captures a face photo, computes an embedding on-device (MobileFaceNet TFLite), and sends it to the backend.
- FastAPI matches the embedding, increments visits, and grants a reward every 5 visits (free coffee).

## Project structure
- `lib/` – Flutter app
- `assets/models/` – TFLite model (MobileFaceNet)
- `restaurant_server/` – FastAPI backend (SQLite)

## Requirements
- Flutter SDK (3.x)
- Android Studio + Android Emulator (or a real Android device)
- Python 3.10+ (recommended)

---

