"""
Módulo de evaluación de modelos
"""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo con múltiples métricas
    
    Args:
        model: Modelo entrenado
        X_test: Features de prueba
        y_test: Target de prueba
        
    Returns:
        Dict con las métricas calculadas
    """
    logger.info("Evaluando modelo...")
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary')
    }
    
    logger.info("=== Métricas del Modelo ===")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Reporte de clasificación
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['Bajo', 'Alto']))
    
    return metrics
