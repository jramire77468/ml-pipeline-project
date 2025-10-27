# 🍷 Wine Quality ML Pipeline con MLflow y CI/CD

Pipeline automatizado de Machine Learning para clasificación de calidad de vino, con tracking en MLflow y CI/CD en GitHub Actions.

## 📋 Descripción

Proyecto MLOps completo que clasifica la calidad del vino usando Random Forest y MLflow para tracking.

## 🚀 Instalación Rápida
```bash
# Crear entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install --upgrade pip setuptools wheel
pip install pandas numpy scikit-learn mlflow pyyaml pytest flake8 requests

# Entrenar modelo
python src/train.py

# Ver resultados
mlflow ui
```

## 📊 Dataset

**Wine Quality Dataset** de UCI ML Repository
- 1,599 muestras de vino tinto portugués
- 11 features fisicoquímicas
- Clasificación binaria: Alta/Baja calidad

## 🏗️ Estructura
```
ml-pipeline-project/
├── src/
│   ├── train.py          # Pipeline principal
│   ├── preprocess.py     # Preprocesamiento
│   └── evaluate.py       # Evaluación
├── tests/
│   └── test_pipeline.py  # Tests
├── .github/workflows/
│   └── ml.yml            # CI/CD
├── config.yaml           # Configuración
└── requirements.txt      # Dependencias
```

## 📈 Uso
```bash
# Entrenar
python src/train.py

# Tests
pytest tests/ -v

# MLflow UI
mlflow ui
```

## ✅ Características

- ✅ Dataset externo (UCI)
- ✅ Preprocesamiento completo
- ✅ Modelo RandomForest
- ✅ 4 métricas (Accuracy, F1, Precision, Recall)
- ✅ MLflow tracking completo
- ✅ CI/CD con GitHub Actions
- ✅ Tests automatizados

## 🎯 Resultados Esperados

- Accuracy: ~0.75-0.85
- F1-Score: ~0.70-0.80

## 👨‍💻 Autor

[Juan Felipe Ramirez Botero]
