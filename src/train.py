import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model, save_model
from src.data_preprocessing import load_and_prepare_data


def train_model(dataframe_direction, epochs=10, batch_size=32):
    """
    Train the bone age prediction model.
    
    Args:
        dataframe_direction: Path to dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("=" * 60)
    print("BONE AGE PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Check GPU availability
    print(f"\nGPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    
    # Load data
    print("\n[1/7] Loading data...")
    train_full_df, test_df = load_and_prepare_data(dataframe_direction)
    
    # Split data
    print("\n[2/7] Splitting data...")
    train_df, valid_df = train_test_split(
        train_full_df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(valid_df)} samples")
    
    # Data augmentation
    print("\n[3/7] Setting up data augmentation...")
    data_augmentation = dict(
        rotation_range=20,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        shear_range=0.05,
        fill_mode="nearest"
    )
    
    train_generator = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_input,
        **data_augmentation
    )
    
    valid_generator = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_input
    )
    
    img_size = (256, 256)
    
    train_data = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col="path",
        y_col="boneage",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="other",
        color_mode="rgb",
        target_size=img_size
    )
    
    valid_data = valid_generator.flow_from_dataframe(
        dataframe=valid_df,
        x_col="path",
        y_col="boneage",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="other",
        color_mode="rgb",
        target_size=img_size
    )
    
    # Build model
    print("\n[4/7] Building model...")
    model = build_model(input_shape=(256, 256, 3))
    model.summary()
    
    # Setup callbacks
    print("\n[5/7] Setting up callbacks...")
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'best_bone_age_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [reduce_lr, checkpoint]
    
    # Train model
    print(f"\n[6/7] Training model for {epochs} epochs...")
    print("=" * 60)
    
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=valid_data,
        callbacks=callbacks,
        verbose=1
    )
    
    print("=" * 60)
    print("Training completed!")
    
    # Save model
    print("\n[7/7] Saving final model...")
    os.makedirs("saved_models", exist_ok=True)
    
    save_model(model, "saved_models/bone_age_model.keras")
    save_model(model, "saved_models/bone_age_model.h5")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    # For local training (update path as needed)
    # dataframe_direction = "path/to/dataset"
    
    # For Kaggle
    dataframe_direction = "/kaggle/input/rsna-bone-age"
    
    model, history = train_model(dataframe_direction, epochs=10, batch_size=32)
