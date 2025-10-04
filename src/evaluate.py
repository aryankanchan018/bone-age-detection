import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_model(model, test_data, verbose=True):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained Keras model
        test_data: Test data generator
        verbose: Print results
    
    Returns:
        Dictionary of evaluation metrics
    """
    if verbose:
        print("Evaluating model on validation data...")
    
    results = model.evaluate(test_data, verbose=1 if verbose else 0)
    
    metrics = {
        'loss': results[0],
        'mae': results[1],
        'mse': results[2],
        'rmse': np.sqrt(results[2])
    }
    
    if verbose:
        print(f"\nValidation Results:")
        print(f"  Loss (MSE): {metrics['loss']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f} months")
        print(f"  RMSE: {metrics['rmse']:.2f} months")
    
    return metrics
