.PHONY: install train test api dashboard docker-build docker-up clean

install:
	pip install -r requirements.txt

train:
	python scripts/train_pipeline.py --n-trials 20

train-fast:
	python scripts/train_pipeline.py --n-trials 5

test:
	pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

test-unit:
	pytest tests/test_data_loader.py tests/test_preprocessor.py -v

test-api:
	pytest tests/test_api.py -v

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py --server.port 8501

docker-build:
	docker compose build

docker-train:
	docker compose --profile train run train

docker-up:
	docker compose up api dashboard

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

setup: install train test
	@echo "✅ Setup complete! Run 'make api' and 'make dashboard' to start."
