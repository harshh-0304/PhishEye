.PHONY: help install test app cli lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r PhishEye/requirements.txt

test:  ## Run the test suite
	cd PhishEye && pytest -q

app:  ## Launch the Streamlit web app
	streamlit run PhishEye/App.py

cli:  ## Run the PhishEye CLI (pass URL=... )
	cd PhishEye && python cli.py $(URL)

lint:  ## Byte-compile all sources to catch syntax errors
	python -m compileall -q PhishEye/phisheye

clean:  ## Remove Python caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
