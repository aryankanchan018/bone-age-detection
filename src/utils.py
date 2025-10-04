import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input


def plot_training_history(history):
    """
    Plot training and validation metrics.
    
    Args:
        history: Keras training history object
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_title('Mean Absolute Error Over Epochs', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (months)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def predict_bone_age(img_path, model):
    """
    Predict bone age from image path.
    
    Args:
        img_path: Path to image file
        model: Trained Keras model
    
    Returns:
        Predicted bone age in months
    """
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array / 255.0)
    
    prediction = model.predict(img_array, verbose=0)
    return prediction[0][0]
