# Adaptive-IPI Demo

This is a production-ready, pure-inference deployment demo for the Adaptive-IPI student model.

The demo consists of two components:
1. **Backend:** A FastAPI server that loads the trained `ModernBERT` checkpoint into memory and exposes an efficient `/predict` API.
2. **Frontend:** A minimalist, responsive React 18 + Vite interface.

## 1. Project Structure
```text
demo/
├── backend/          # FastAPI inference server
├── frontend/         # React + Vite UI
├── models/           # Stores the exported student model checkpoint
├── docker-compose.yml
└── README.md
```

## 2. Local Development (No Docker)

### Starting the Backend
1. Ensure your model is copied to `demo/models/adaptive_ipi_model`.
2. Activate your Python virtual environment.
3. Install backend requirements:
   ```bash
   cd demo/backend
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Starting the Frontend
1. Install Node.js (v18+).
2. Install frontend dependencies:
   ```bash
   cd demo/frontend
   npm install
   ```
3. Run the Vite development server:
   ```bash
   npm run dev
   ```
4. Open the displayed local URL (usually `http://localhost:5173`) in your browser.

## 3. Production Deployment (Docker)

To deploy the demo in a containerized environment:

1. Build the frontend static assets:
   ```bash
   cd demo/frontend
   npm run build
   ```
2. Run Docker Compose from the `demo/` directory:
   ```bash
   docker-compose up --build -d
   ```
The backend API will be available at `http://localhost:8000/`. (To serve the frontend in production, you can either configure an Nginx proxy container or mount the `frontend/dist` folder into the FastAPI `StaticFiles` router in `main.py`).
