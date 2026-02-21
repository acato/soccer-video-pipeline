.PHONY: help setup deploy test-unit test-integration test-e2e up down logs generate-fixtures check-nas check-gpu

help:
	@echo "Soccer Video Processing Pipeline"
	@echo ""
	@echo "  make deploy             Auto-detect hardware, generate .env, start stack"
	@echo "  make setup              Install Python dependencies (local dev)"
	@echo "  make generate-fixtures  Generate synthetic test video fixtures"
	@echo "  make test-unit          Run unit tests (no infra required)"
	@echo "  make test-integration   Run integration tests (requires Docker)"
	@echo "  make up                 Start stack (assumes .env exists)"
	@echo "  make down               Stop stack"
	@echo "  make logs               Tail worker + api logs"
	@echo "  make check-nas          Run NAS health check"
	@echo "  make check-gpu          Check NVIDIA GPU + container toolkit"

deploy:
	infra/scripts/setup.sh

setup:
	pip install -r requirements.txt

generate-fixtures:
	python tests/fixtures/fixture_generator.py

test-unit:
	pytest tests/unit/ -m unit -v --cov=src --cov-report=term-missing

test-integration:
	docker-compose -f infra/docker-compose.test.yml up -d
	sleep 3
	pytest tests/integration/ -m integration -v
	docker-compose -f infra/docker-compose.test.yml down

test-e2e:
	docker-compose -f infra/docker-compose.yml up -d
	sleep 10
	pytest tests/e2e/ -m e2e -v --timeout=600
	docker-compose -f infra/docker-compose.yml down

up:
	cp -n infra/.env.example infra/.env || true
	docker-compose -f infra/docker-compose.yml --env-file infra/.env up -d
	@echo "API: http://localhost:8080"
	@echo "Flower (job monitor): http://localhost:5555"

down:
	docker-compose -f infra/docker-compose.yml down

logs:
	docker-compose -f infra/docker-compose.yml logs -f worker api

check-nas:
	infra/scripts/check_nas.sh

check-gpu:
	infra/scripts/check_gpu.sh
