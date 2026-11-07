# FaceTrack — from repo root. Each target prints then runs.

.PHONY: install install-backend install-frontend \
	db run fresh api web build-web test check count rtsp

# One-shot: Python (tracker + API) + frontend npm deps
install: install-backend install-frontend
	@echo done: make install

install-backend:
	@echo pip install -r backend/requirements.txt -r backend/requirements-api.txt
	pip install -r backend/requirements.txt -r backend/requirements-api.txt

install-frontend:
	@echo cd frontend \&\& npm install
	cd frontend && npm install

db:
	@echo docker compose -f deploy/docker-compose.yml up -d db
	docker compose -f deploy/docker-compose.yml up -d db

run:
	@echo python main.py
	python main.py

fresh:
	@echo python main.py --fresh
	python main.py --fresh

api:
	@echo uvicorn api.server:app --app-dir backend --port 8000
	uvicorn api.server:app --app-dir backend --port 8000

web:
	@echo cd frontend \&\& npm run dev
	cd frontend && npm run dev

build-web:
	@echo cd frontend \&\& npm run build
	cd frontend && npm run build

test:
	@echo pytest -q
	pytest -q

check:
	@echo python backend/scripts/final_check.py
	python backend/scripts/final_check.py

count:
	@echo python backend/scripts/count.py --source both --session latest
	python backend/scripts/count.py --source both --session latest

rtsp:
	@echo python backend/scripts/run_rtsp.py
	python backend/scripts/run_rtsp.py
