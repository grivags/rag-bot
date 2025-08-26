# RAG Bot — FastAPI + LangChain + Chroma + Mistral 7B (OpenRouter)

Mini RAG end-to-end demo project:

- Ingest: PDF/TXT → split → embed (HuggingFace `all-MiniLM-L6-v2`) → Chroma  
- API: FastAPI `/ask` (RAG endpoint), CORS enabled, optional `/reload`  
- LLM: `mistralai/mistral-7b-instruct:free` via OpenRouter (OpenAI-compatible API)

---

## 🚀 Quickstart (Windows)

1. Clone & masuk folder  
   ```
   git clone https://github.com/<username>/rag-bot.git
   cd rag-bot
   ```

2. Bikin virtual environment  
   ```
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies  
   ```
   pip install -r requirements.txt
   ```

4. Copy env template  
   ```
   copy .env.example .env
   ```
   → isi API key di file `.env` (OpenRouter atau OpenAI)

5. Ingest dokumen (taruh PDF/TXT di folder `docs/`)  
   ```
   python ingest.py
   ```

6. Run server  
   ```
   uvicorn app:app --reload --port 8000
   ```

---

## 📡 API Endpoints

- `GET /` → health check  
- `POST /ask` → query ke dokumen  
  Example body:
  ```json
  { "question": "Ringkas isi dokumen" }
  ```
- (opsional) `POST /reload` → reload Chroma DB tanpa restart server

---

## 🖥️ Test cepat (PowerShell)

```
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post `
  -ContentType "application/json" -Body (@{question="Ringkas isi dokumen"} | ConvertTo-Json)
```

---

## 🌐 Optional UI

Ada `static.html` sederhana buat ngetes via browser.

```
py -m http.server 5500
```

Buka: [http://localhost:5500/static.html](http://localhost:5500/static.html)

---

## 📂 Project Structure

```
rag-bot/
├─ docs/          # PDF/TXT source docs
├─ chroma_db/     # auto-generated vector store
├─ app.py         # FastAPI app (RAG backend)
├─ ingest.py      # ingest pipeline (build vector DB)
├─ static.html    # optional mini UI
├─ requirements.txt
├─ .env.example   # template env
├─ .gitignore
└─ README.md
```

---

## ⚠️ Notes

- Jangan commit file `.env` dan `chroma_db/` (sudah di-ignore).  
- Model default: `mistralai/mistral-7b-instruct:free` (via OpenRouter).  
- Bisa ganti model via `LLM_MODEL` di `.env`.

---

## 📜 License

MIT License © 2025  
