# Rappi Ops Intelligence — Web App 🚀

Dashboard de inteligencia operativa con chat en lenguaje natural, visualizaciones automáticas y detección de insights. Construido con FastAPI + vanilla JS.

## Estructura

```
rappi_web/
├── app.py              # FastAPI backend (API + servidor de frontend)
├── data_loader.py      # Transformación wide→long + feature engineering
├── query_engine.py     # NL → consultas pandas
├── insights.py         # Motor de detección automática
├── static/
│   └── index.html      # Frontend completo (chat + charts + insights)
├── data.xlsx           # Tu archivo de datos (no se sube a GitHub)
├── requirements.txt
├── .env.example
└── README.md
```

## Instalación local

```bash
# 1. Clonar y entrar
git clone https://github.com/CamiloC0rtes/rappi-ops-intelligence.git
cd rappi-ops-intelligence

# 2. Entorno virtual
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate.ps1

# 3. Dependencias
pip install -r requirements.txt

# 4. Variables de entorno
cp .env.example .env
# Edita .env → OPENAI_API_KEY=sk-...

# 5. Coloca tu Excel como data.xlsx en la raíz

# 6. Iniciar
python app.py
# → Abre http://localhost:8000
```

## Despliegue en Railway

1. Sube a GitHub (sin `data.xlsx` y sin `.env`)
2. Railway → **New Project → Deploy from GitHub**
3. Agrega variables de entorno:
   - `OPENAI_API_KEY` = tu key
   - `PORT` = 8000 (Railway lo setea automáticamente)
4. Start command: `python app.py`
5. ✅ Railway detecta `requirements.txt` y despliega automáticamente

> Para Railway necesitarás subir `data.xlsx` de alguna forma (volumen, o convertirlo a un CSV embebido en el repo).

## API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/api/chat` | Chat con el bot |
| `GET`  | `/api/insights` | Lista de insights (filtrable) |
| `GET`  | `/api/insights/summary` | Resumen por tipo/severidad/país |
| `GET`  | `/api/filters` | Valores disponibles para filtros |
| `GET`  | `/api/ranking` | Top/bottom zonas por métrica |
| `GET`  | `/api/timeseries` | Serie de tiempo de una zona |
| `DELETE` | `/api/session/{id}` | Resetear sesión de chat |

## .gitignore

```
.env
data.xlsx
*.xlsx
__pycache__/
.venv/
*.pyc
.DS_Store
```
