.PHONY: help up down logs build rebuild test lint format psql clean

help:
	@echo "Targets:"
	@echo "  up         start postgres + streamlit"
	@echo "  down       stop containers"
	@echo "  logs       tail logs"
	@echo "  build      build image"
	@echo "  rebuild    rebuild image without cache"
	@echo "  test       run pytest suite"
	@echo "  lint       ruff + mypy"
	@echo "  format     ruff format + autofix"
	@echo "  psql       open psql shell"
	@echo "  clean      remove containers + volumes"

up:
	docker compose up -d
	@echo "Streamlit: http://localhost:8501"

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

build:
	docker compose build

rebuild:
	docker compose build --no-cache

test:
	docker compose exec app pytest -v

lint:
	docker compose exec app ruff check src tests
	docker compose exec app mypy src

format:
	docker compose exec app ruff format src tests
	docker compose exec app ruff check --fix src tests

psql:
	docker compose exec postgres psql -U agent

clean:
	docker compose down -v
