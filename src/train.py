"""
Script principal de entrenamiento del pipeline ML
"""
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import load_data, clean_data, prepare_features, split_data
from src.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    """Carga la configuración desde el archivo YAML"""
    logger.info(f"Cargando configuración desde: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Función principal del pipeline"""
    
    config = load_config()
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    logger.info("="*50)
    logger.info("INICIANDO PIPELINE DE MACHINE LEARNING")
    logger.info("="*50)
    
    with mlflow.start_run():
        
        logger.info("\n[1/5] CARGA DE DATOS")
        df = load_data(config['data']['url'])
        
        mlflow.log_param("dataset_source", config['data']['url'])
        mlflow.log_param("dataset_rows", df.shape[0])
        mlflow.log_param("dataset_cols", df.shape[1])
        
        logger.info("\n[2/5] LIMPIEZA DE DATOS")
        df_clean = clean_data(df, config['preprocessing']['handle_missing'])
        
        logger.info("\n[3/5] PREPARACIÓN DE FEATURES")
        X, y, scaler = prepare_features(
            df_clean,
            config['data']['target_column'],
            config['preprocessing']['scale']
        )
        
        X_train, X_test, y_train, y_test = split_data(
            X, y,
            config['data']['test_size'],
            config['data']['random_state']
        )
        
        mlflow.log_param("test_size", config['data']['test_size'])
        mlflow.log_param("scaling", config['preprocessing']['scale'])
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        
        logger.info("\n[4/5] ENTRENAMIENTO DEL MODELO")
        model_params = config['model']['params']
        
        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)
        
        model = RandomForestClassifier(**model_params)
        logger.info(f"Entrenando {config['model']['name']}...")
        model.fit(X_train, y_train)
        logger.info("✓ Modelo entrenado exitosamente")
        
        logger.info("\n[5/5] EVALUACIÓN DEL MODELO")
        metrics = evaluate_model(model, X_test, y_test)
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        logger.info("\nREGISTRANDO MODELO EN MLFLOW")
        
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(1)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="wine-quality-classifier"
        )
        
        logger.info("✓ Modelo registrado exitosamente en MLflow")
        
        if scaler:
            import joblib
            scaler_path = "scaler.pkl"
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path)
            os.remove(scaler_path)
            logger.info("✓ Scaler guardado como artefacto")
        
        logger.info("\n" + "="*50)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("="*50)
        logger.info(f"\nMétricas finales:")
        logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  - F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  - Precision: {metrics['precision']:.4f}")
        logger.info(f"  - Recall: {metrics['recall']:.4f}")
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"\nMLflow Run ID: {run_id}")
        logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
