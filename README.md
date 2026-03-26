# comp5523 — hand / bottle vision + voice intent

## How to run

1. **Python**  
   Use Python **3.10+** (3.11 or 3.12 is fine).

2. **Create a virtual environment** (recommended)

   ```bash
   cd /path/to/comp5523
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **macOS:** If `PyAudio` fails with `portaudio.h` not found, install PortAudio first, then retry:

   ```bash
   brew install portaudio
   pip install pyaudio
   ```

4. **Run the app**

   ```bash
   python main.py
   ```

5. **First launch**  
   - The first run may download **`hand_landmarker.task`** (MediaPipe) and **`yolov8n.pt`** (Ultralytics YOLO).  
   - The first time you use speech intent, **SentenceTransformer** may download **`all-MiniLM-L6-v2`**.  
   These need a working internet connection.

6. **Use the windows**  
   - Focus the **Webcam** window so keyboard shortcuts work.  
   - **`S`** — listen once, transcribe (Google), classify intent, and show a bottle-position answer when the intent is “where is the bottle”.  
   - **`Q`** — quit.

7. **Permissions (macOS)**  
   Allow the terminal or IDE to use the **camera** and **microphone** under **System Settings → Privacy & Security**.

---

Speech recognition uses Google’s web API (requires network). Semantic intent uses **sentence-transformers** locally after the model is cached.

