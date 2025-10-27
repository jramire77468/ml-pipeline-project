.PHONY: install train test lint clean mlflow-ui help

help:
    @echo "Comandos disponibles:"
    @echo "  make install    - Instala las dependencias"
    @echo "  make train      - Ejecuta el pipeline de entrenamiento"
    @echo "  make test       - Ejecuta los tests"
    @echo "  make lint       - Ejecuta el linter (flake8)"
    @echo "  make mlflow-ui  - Inicia la interfaz de MLflow"
    @echo "  make clean      - Limpia archivos temporales"

install:
    @echo "📦 Instalando dependencias..."
    pip install --upgrade pip
    pip install -r requirements.txt
    @echo "✓ Dependencias instaladas"

train:
    @echo "🚀 Ejecutando pipeline de ML..."
    python src/train.py
    @echo "✓ Pipeline completado"

test:
    @echo "🧪 Ejecutando tests..."
    pytest tests/ -v --tb=short
    @echo "✓ Tests completados"

lint:
    @echo "🔍 Ejecutando linter..."
    flake8 src/ --max-line-length=100 --ignore=E501,W503
    @echo "✓ Linting completado"

mlflow-ui:
    @echo "🎨 Iniciando MLflow UI en http://localhost:5000"
    mlflow ui

clean:
    @echo "🧹 Limpiando archivos temporales..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    @echo "✓ Limpieza completada"
