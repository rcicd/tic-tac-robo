.PHONY: setup install run clean

PYTHON_VERSION = 3.8.18
VENV_NAME = venv
WHEEL_NAME = kortex_api-2.6.0.post3-py3-none-any.whl

PYTHON_BIN = $(HOME)/.pyenv/versions/$(VENV_NAME)/bin/python
PIP_BIN = $(HOME)/.pyenv/versions/$(VENV_NAME)/bin/pip

setup:
	pyenv install -s $(PYTHON_VERSION)
	pyenv virtualenv --force $(PYTHON_VERSION) $(VENV_NAME)
	pyenv local $(VENV_NAME)

install: setup
	@echo "📦 Installing requirements.txt..."
	$(PIP_BIN) install -r requirements.txt

	@if [ ! -f $(WHEEL_NAME) ]; then \
		echo "🚫 Wheel '$(WHEEL_NAME)' not found!"; \
		exit 1; \
	fi

	@echo "🔧 Installing wheel..."
	$(PIP_BIN) install $(WHEEL_NAME)

run:
	$(PYTHON_BIN) src/ui.py

clean:
	rm -f .python-version
	pyenv uninstall -f $(VENV_NAME)
