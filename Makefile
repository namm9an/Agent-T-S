# Makefile for LangGraph-Orchestrated Transcription + Summarization Service

.PHONY: help setup install dev start-langgraph start-api test clean logs

# Variables
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)LangGraph Transcription + Summarization Service$(NC)"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Complete setup (venv, dependencies, directories, env file)
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@echo "Upgrading pip..."
	@$(PIP) install --upgrade pip wheel setuptools
	@echo "Installing dependencies..."
	@$(PIP) install -r requirements.txt
	@echo "Creating required directories..."
	@mkdir -p jobs uploads logs
	@if [ ! -f .env ]; then \
		echo "Creating .env file from template..."; \
		cp .env.example .env; \
		echo "$(YELLOW)Please edit .env and add your API keys$(NC)"; \
	fi
	@echo "Making scripts executable..."
	@chmod +x scripts/*.sh 2>/dev/null || true
	@echo "$(GREEN)Setup complete! Run 'make dev' to start services$(NC)"

install: ## Install Python dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

dev: ## Start both LangGraph and FastAPI servers (development mode)
	@echo "$(GREEN)Starting development servers...$(NC)"
	@echo "Starting LangGraph server on port 8123..."
	@./scripts/start_langgraph.sh &
	@sleep 3
	@echo "Starting FastAPI server on port 8000..."
	@./scripts/start_api.sh

start-langgraph: ## Start only LangGraph server
	@echo "$(GREEN)Starting LangGraph server...$(NC)"
	@./scripts/start_langgraph.sh

start-api: ## Start only FastAPI server
	@echo "$(GREEN)Starting FastAPI server...$(NC)"
	@./scripts/start_api.sh

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	@$(PYTHON_VENV) -m pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	@$(PYTHON_VENV) -m pytest tests/test_app.py -v

test-integration: ## Run integration tests (requires services running)
	@echo "$(GREEN)Running integration tests...$(NC)"
	@$(PYTHON_VENV) -m pytest tests/integration_local.py -v

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@$(PYTHON_VENV) -m pytest --cov=app --cov-report=term-missing tests/

lint: ## Run code linting
	@echo "$(GREEN)Running code linting...$(NC)"
	@$(PYTHON_VENV) -m flake8 app/ tests/ || true
	@$(PYTHON_VENV) -m black --check app/ tests/ || true

format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	@$(PYTHON_VENV) -m black app/ tests/ *.py

check-env: ## Check environment variables
	@echo "$(GREEN)Checking environment configuration...$(NC)"
	@$(PYTHON_VENV) -c "import os; \
		vars = ['LANGGRAPH_ENDPOINT', 'QWEN_ENDPOINT', 'WHISPER_ENDPOINT']; \
		for v in vars: \
			val = os.getenv(v, 'NOT SET'); \
			print(f'{v}: {val[:50]}...' if len(val) > 50 else f'{v}: {val}')"

health: ## Check service health
	@echo "$(GREEN)Checking service health...$(NC)"
	@echo "FastAPI Health:"
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "$(RED)FastAPI not responding$(NC)"
	@echo ""
	@echo "LangGraph Health:"
	@curl -s http://localhost:8123/health | python3 -m json.tool || echo "$(RED)LangGraph not responding$(NC)"

logs: ## Show service logs
	@echo "$(GREEN)Showing recent logs...$(NC)"
	@tail -n 50 logs/*.log 2>/dev/null || echo "No log files found"

logs-follow: ## Follow service logs in real-time
	@echo "$(GREEN)Following logs (Ctrl+C to exit)...$(NC)"
	@tail -f logs/*.log 2>/dev/null || echo "No log files found"

clean: ## Clean up generated files and caches
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf *.egg-info
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)Cleanup complete$(NC)"

clean-all: clean ## Clean everything including jobs and uploads
	@echo "$(YELLOW)Removing job files...$(NC)"
	@rm -rf jobs/*.json
	@echo "$(YELLOW)Removing uploaded files...$(NC)"
	@rm -rf uploads/*
	@echo "$(GREEN)Full cleanup complete$(NC)"

reset: clean-all ## Reset to fresh state (removes venv too)
	@echo "$(RED)Removing virtual environment...$(NC)"
	@rm -rf $(VENV)
	@echo "$(RED)Removing .env file...$(NC)"
	@rm -f .env
	@echo "$(GREEN)Reset complete. Run 'make setup' to start fresh$(NC)"

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	@docker build -t langgraph-transcription:latest .

docker-run: ## Run services in Docker
	@echo "$(GREEN)Starting services in Docker...$(NC)"
	@docker-compose up -d

docker-stop: ## Stop Docker services
	@echo "$(YELLOW)Stopping Docker services...$(NC)"
	@docker-compose down

docker-logs: ## Show Docker service logs
	@docker-compose logs -f

# Quick test commands
test-transcribe: ## Test transcription endpoint
	@echo "$(GREEN)Testing transcription...$(NC)"
	@echo "Creating test audio file..."
	@echo "test" > test.txt && echo "Test file created as placeholder"
	@curl -X POST "http://localhost:8000/transcribe?wait=true" \
		-F "file=@test.txt" \
		-F "diarize=false" | python3 -m json.tool

test-summarize: ## Test summarization endpoint
	@echo "$(GREEN)Testing summarization...$(NC)"
	@curl -X POST "http://localhost:8000/summarize?wait=true" \
		-H "Content-Type: application/json" \
		-d '{"text": "This is a test text for summarization.", "summary_type": "executive"}' \
		| python3 -m json.tool

test-callback: ## Test callback endpoint
	@echo "$(GREEN)Testing callback...$(NC)"
	@curl -X POST "http://localhost:8000/callbacks/langgraph" \
		-H "X-LangGraph-Secret: dev-secret-key" \
		-H "Content-Type: application/json" \
		-d '{"job_id": "test-123", "status": "completed", "result": {"transcript": "Test"}}' \
		| python3 -m json.tool

# Deployment helpers
deploy-e2e: ## Deploy to E2E Networks (requires SSH config)
	@echo "$(GREEN)Deploying to E2E Networks...$(NC)"
	@echo "This would deploy to your E2E node"
	@echo "Configure SSH and run: ssh user@e2e-node 'cd /path/to/app && git pull && make install'"

# Default target
.DEFAULT_GOAL := help
