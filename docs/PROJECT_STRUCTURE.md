# Project Structure - Agent T+S

## 📁 Directory Organization

```
Agent T+S/
├── app/                      # Core application code
│   ├── __init__.py
│   ├── main.py              # FastAPI main application
│   ├── schemas.py           # Data models and schemas  
│   ├── integrations/        # External service integrations
│   │   ├── __init__.py
│   │   ├── langgraph.py     # LangGraph integration
│   │   └── remote_client.py # Remote API client
│   ├── services/            # Business logic
│   │   ├── __init__.py
│   │   └── storage.py       # Job storage service
│   └── utils/               # Utility functions
│       ├── __init__.py
│       └── logger.py        # Logging configuration
│
├── scripts/                 # Utility and helper scripts
│   ├── mock_langgraph.py   # Mock LangGraph for testing
│   └── qwen_check.py        # Qwen model connectivity check
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_app.py         # Main application tests
│   ├── test_direct_whisper.py  # Whisper API tests
│   ├── test_new_whisper.py     # Updated Whisper tests
│   └── test_speed.py           # Performance tests
│
├── data/                    # Runtime data (gitignored)
│   ├── jobs/               # Job JSON files
│   ├── uploads/            # Uploaded audio files
│   └── logs/               # Application logs
│
├── config/                  # Configuration files
│   └── .env.example        # Environment template
│
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md     # System architecture
│   └── DEPLOY.md          # Deployment guide
│
├── streamlit_app.py        # Streamlit UI application
├── start_services.sh       # Main startup script
├── start.sh               # Alternative startup
├── requirements.txt        # Python dependencies
├── Makefile               # Build automation
├── README.md              # Project documentation
├── project.md             # Project overview
├── .gitignore             # Git ignore rules
└── .env                   # Local environment (gitignored)
```

## 🎯 Core Components

### **Main Application (`streamlit_app.py`)**
- Streamlit-based UI for audio transcription and summarization
- Integrates with FastAPI backend
- Provides real-time job status tracking

### **FastAPI Backend (`app/`)**
- RESTful API for transcription and summarization
- Manages job lifecycle
- Integrates with LangGraph for workflow orchestration

### **Services**
- **Storage Service**: Manages job persistence and retrieval
- **LangGraph Integration**: Handles workflow triggers
- **Remote Client**: Manages external API communications

## 🚀 Running the Application

### Quick Start
```bash
# Using the startup script
./start_services.sh

# Or using Make
make dev
```

### Individual Services
```bash
# Start Streamlit UI
streamlit run streamlit_app.py

# Start FastAPI backend  
uvicorn app.main:app --reload --port 8000
```

## 📝 Key Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main UI application |
| `app/main.py` | FastAPI endpoints |
| `app/schemas.py` | Data models |
| `app/integrations/langgraph.py` | Workflow orchestration |
| `start_services.sh` | Unified startup script |
| `requirements.txt` | Python dependencies |

## 🧹 Maintenance

### Runtime Data
All runtime data is stored in the `data/` directory:
- Job files: `data/jobs/`
- Uploaded files: `data/uploads/`
- Logs: `data/logs/`

This directory is gitignored to prevent committing sensitive or temporary data.

### Testing
Run tests with:
```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_app.py
```

## 🔒 Security Notes

- Never commit `.env` files
- API keys should be stored in environment variables
- The `data/` directory contains potentially sensitive information and is gitignored
- Use `.env.example` as a template for environment configuration

## 📚 Documentation

- `README.md` - Main project documentation
- `docs/ARCHITECTURE.md` - System design and architecture
- `docs/DEPLOY.md` - Deployment instructions
- `PROJECT_STRUCTURE.md` - This file

## 🛠️ Development Workflow

1. **Environment Setup**: Copy `config/.env.example` to `.env` and configure
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run Tests**: `pytest tests/`
4. **Start Services**: `./start_services.sh`
5. **Access UI**: Open http://localhost:8501
