# AgriTrust: An Evidence- and Confidence-Aware Multi-Agent RAG Framework
PYTHON = .\.venv\python.exe

.PHONY: all setup corpus index test eval app api clean

all: setup corpus index test eval

setup:
	$(PYTHON) -m pip install -r requirements.txt

corpus:
	$(PYTHON) scripts/build_corpus.py

index:
	$(PYTHON) scripts/build_index.py

test:
	$(PYTHON) -m pytest -v tests/

eval:
	$(PYTHON) eval/generate_dataset.py
	$(PYTHON) eval/tune_thresholds.py
	$(PYTHON) eval/run_eval.py --split test

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

clean:
	rm -rf .cache results/*.png results/*.csv results/*.md
