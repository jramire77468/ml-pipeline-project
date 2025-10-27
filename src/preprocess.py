"""
Módulo de preprocesamiento de datos
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(url: str) -> pd.DataFrame:
    """
    Carga datos desde una URL
    
    Args:
        url: URL del dataset
        
    Returns:
        DataFrame con los datos cargados
    """
    logger.info(f"Cargando datos desde: {url}")
    try:
        df = pd.read_csv(url, sep=';')
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        logger.info("Intentando con encoding diferente...")
        try:
            df = pd.read_csv(url, sep=';', encoding='latin-1')
            logger.info(f"Datos cargados con latin-1: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
        except Exception as e2:
            logger.error(f"Error al cargar datos con latin-1: {e2}")
            raise


def clean_data(df: pd.DataFrame, handle_missing: bool = True) -> pd.DataFrame:
    """
    Limpia el dataset
    
    Args:
        df: DataFrame original
        handle_missing: Si se deben manejar valores faltantes
        
    Returns:
        DataFrame limpio
    """
    logger.info("Iniciando limpieza de datos...")
    
    # Eliminar duplicados
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    logger.info(f"Duplicados eliminados: {initial_rows - df.shape[0]}")
    
    # Manejar valores faltantes
    if handle_missing:
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        logger.info(f"Valores faltantes eliminados: {missing_before}")
    
    return df


def prepare_features(df: pd.DataFrame, target_column: str, scale: bool = True):
    """
    Prepara features y target, y realiza el split
    
    Args:
        df: DataFrame procesado
        target_column: Nombre de la columna objetivo
        scale: Si se debe escalar las features
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    logger.info("Preparando features y target...")
    
    # Convertir calidad a binario (bueno >= 6, malo < 6)
    df['quality_binary'] = (df[target_column] >= 6).astype(int)
    
    # Separar features y target
    X = df.drop([target_column, 'quality_binary'], axis=1)
    y = df['quality_binary']
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Distribución de clases: {y.value_counts().to_dict()}")
    
    # Escalar features
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        logger.info("Features escaladas con StandardScaler")
    
    return X, y, scaler


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Divide los datos en train y test
    
    Args:
        X: Features
        y: Target
        test_size: Proporción del conjunto de prueba
        random_state: Semilla aleatoria
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Dividiendo datos: {test_size*100}% test, {(1-test_size)*100}% train")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} muestras")
    logger.info(f"Test set: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test
