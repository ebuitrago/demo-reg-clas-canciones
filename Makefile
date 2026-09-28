# Atajos para Linux, macOS y Codespaces. En Windows, ejecute directamente los comandos de cada regla.
PYTHON ?= python
URL ?= http://127.0.0.1:8000

.PHONY: instalar regresion clasificacion probar lint servir consumir humo limpiar

instalar:  ## Instala dependencias de desarrollo
	$(PYTHON) -m pip install -r requirements-dev.txt

regresion:  ## Parte 1: entrena la regresión y escribe models/modelo.joblib
	$(PYTHON) -m src.entrenar_regresion

clasificacion:  ## Parte 2: entrena el clasificador y escribe models/modelo.joblib
	$(PYTHON) -m src.entrenar_clasificacion

probar:  ## Pruebas automáticas
	$(PYTHON) -m pytest

lint:  ## Revisión de estilo y errores comunes
	ruff check .

servir:  ## Levanta el servicio en http://127.0.0.1:8000
	uvicorn src.app.main:app --reload --port 8000

consumir:  ## Lista priorizada contra el servicio en URL
	$(PYTHON) -m src.consumir $(URL)

humo:  ## Prueba de humo contra el servicio en URL
	$(PYTHON) -m src.probar_servicio $(URL)

limpiar:  ## Borra salidas y cachés
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
