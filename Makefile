# FaceTrack — common commands (print then run). Windows: use the same shell lines from README.

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BACKEND := $(ROOT)backend
FRONTEND := $(ROOT)frontend

.PHONY: install run fresh api web build-web test check count rtsp

install:
	@echo pip install -r backend/requirements.txt \&\& pip install -r backend/requirements-api.txt
	pip install -r backend/requirements.txt
	pip install -r backend/requirements-api.txt

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
