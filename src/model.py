import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception


def build_model(input_shape=(256, 256, 3)):
    """
    Build Xception-based bone age prediction model using Functional API.
    
    Args:
        input_shape: Tuple of (height, width, channels)
    
    Returns:
        Compiled Keras model
    """
    # Define input
    inputs = Input(shape=input_shape)
    
    # Load pre-trained Xception without top layers
    base_model = Xception(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # Fine-tune last layers
    base_model.trainable = True
    
    # Build model
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['mae', 'mse']
    )
    
    return model


def save_model(model, path):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained Keras model
        path: Path to save the model
    """
    model.save(path)
    print(f"✓ Model saved to {path}")
