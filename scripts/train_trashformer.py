import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import warnings

# Suppress palette image warnings
warnings.filterwarnings('ignore', message='Palette images with Transparency')


# CONFIGURATION
class Config:
    """Training configuration"""
    # Paths
    TRAIN_DIR = "../waste_data_split/train"
    VAL_DIR = "../waste_data_split/val"
    MODEL_DIR = "../models"
    
    # Image settings
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # Training settings
    EPOCHS = 30
    INITIAL_LR = 0.001
    
    # Model settings
    BASE_MODEL = 'MobileNetV2'
    DENSE_UNITS = 256
    DROPOUT_RATE = 0.4
    
    # Callbacks settings
    EARLY_STOP_PATIENCE = 7
    LR_REDUCE_PATIENCE = 3
    LR_REDUCE_FACTOR = 0.5
    MIN_LR = 1e-7
    
    # Fine-tuning settings
    ENABLE_FINE_TUNING = True
    FINE_TUNE_EPOCHS = 10
    FINE_TUNE_LR = 1e-5
    FINE_TUNE_LAYERS = 20  # Number of layers to unfreeze from the end


# CUSTOM CALLBACKS FOR BETTER LOGGING

class TrainingLogger(Callback):
    """Custom callback for enhanced training progress logging"""
    
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        
    def on_train_begin(self, logs=None):
        print("\n" + "="*70)
        print(f"🚀 STARTING TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print(f"📊 Total Epochs: {self.params['epochs']}")
        print(f"📦 Batch Size: {Config.BATCH_SIZE}")
        print(f"🎯 Training Samples: {self.params['steps'] * Config.BATCH_SIZE}")
        print(f"✅ Validation Samples: {len(val_generator) * Config.BATCH_SIZE}")
        print("="*70 + "\n")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.now()
        print(f"\n{'─'*70}")
        print(f"📍 EPOCH {epoch + 1}/{self.params['epochs']}")
        print(f"{'─'*70}")
    
    def on_epoch_end(self, epoch, logs=None):
        duration = (datetime.now() - self.epoch_start_time).total_seconds()
        
        train_acc = logs.get('accuracy', 0)
        train_loss = logs.get('loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        lr = float(self.model.optimizer.learning_rate)
        
        print(f"\n{'─'*70}")
        print(f"📈 EPOCH {epoch + 1} RESULTS:")
        print(f"   ⏱️  Duration: {duration:.1f}s")
        print(f"   🎓 Training   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"   ✅ Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
        print(f"   📊 Learning Rate: {lr:.2e}")
        print(f"{'─'*70}")
    
    def on_train_end(self, logs=None):
        print("\n" + "="*70)
        print(f"🎉 TRAINING COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")


# DATA PREPROCESSING

def create_data_generators():
    """Create and configure data generators with augmentation"""
    
    print("\n🔧 Setting up data generators...")
    
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load datasets
    train_generator = train_datagen.flow_from_directory(
        Config.TRAIN_DIR,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        Config.VAL_DIR,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ Found {train_generator.samples} training images in {train_generator.num_classes} classes")
    print(f"✅ Found {val_generator.samples} validation images in {val_generator.num_classes} classes")
    
    # Print class distribution
    print("\n📋 Class Distribution:")
    for class_name, class_idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
        train_count = sum([1 for label in train_generator.labels if label == class_idx])
        print(f"   {class_idx}: {class_name:15s} → {train_count:5d} images")
    
    return train_generator, val_generator


# MODEL ARCHITECTURE

def build_model(num_classes):
    """Build and compile the Trashformer model"""
    
    print("\n🏗️  Building model architecture...")
    
    # Load base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*Config.IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='bn_1')(x)
    x = Dropout(Config.DROPOUT_RATE, name='dropout_1')(x)
    x = Dense(Config.DENSE_UNITS, activation='relu', name='dense_1')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(Config.DROPOUT_RATE * 0.5, name='dropout_2')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    optimizer = Adam(learning_rate=Config.INITIAL_LR)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model compiled with {model.count_params():,} parameters")
    print(f"   → Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"   → Non-trainable params: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")
    
    return model, base_model


# CALLBACKS SETUP

def setup_callbacks():
    """Configure training callbacks"""
    
    print("\n⚙️  Setting up callbacks...")
    
    # Create models directory if it doesn't exist
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Timestamp for unique model naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(Config.MODEL_DIR, f'trashformer_best_{timestamp}.keras')
    
    # Model checkpoint - save best model
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=Config.EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=0
    )
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=Config.LR_REDUCE_FACTOR,
        patience=Config.LR_REDUCE_PATIENCE,
        min_lr=Config.MIN_LR,
        verbose=0
    )
    
    # Custom training logger
    training_logger = TrainingLogger()
    
    print(f"✅ Model checkpoint: {model_path}")
    print(f"✅ Early stopping patience: {Config.EARLY_STOP_PATIENCE} epochs")
    print(f"✅ Learning rate reduction: factor={Config.LR_REDUCE_FACTOR}, patience={Config.LR_REDUCE_PATIENCE}")
    
    return [checkpoint, early_stop, reduce_lr, training_logger], model_path


# FINE-TUNING FUNCTION

def fine_tune_model(model, base_model, train_generator, val_generator, initial_history):
    """
    Fine-tune the model by unfreezing top layers of the base model
    
    Args:
        model: Compiled model
        base_model: Base model (MobileNetV2)
        train_generator: Training data generator
        val_generator: Validation data generator
        initial_history: History from initial training
    
    Returns:
        Combined history dictionary
    """
    
    if not Config.ENABLE_FINE_TUNING:
        print("\n⏭️  Fine-tuning is disabled. Skipping...")
        return initial_history
    
    print("\n" + "="*70)
    print("🔧 FINE-TUNING PHASE")
    print("="*70)
    
    # Unfreeze the base model
    print(f"\n📖 Unfreezing top {Config.FINE_TUNE_LAYERS} layers of base model...")
    base_model.trainable = True
    
    # Freeze all layers except the last N
    layers_to_train = []
    for layer in base_model.layers[:-Config.FINE_TUNE_LAYERS]:
        layer.trainable = False
    
    for layer in base_model.layers[-Config.FINE_TUNE_LAYERS:]:
        layer.trainable = True
        layers_to_train.append(layer.name)
    
    print(f"✅ Unfrozen {Config.FINE_TUNE_LAYERS} layers:")
    for i, name in enumerate(layers_to_train[:5], 1):
        print(f"   {i}. {name}")
    if len(layers_to_train) > 5:
        print(f"   ... and {len(layers_to_train) - 5} more")
    
    # Recompile with lower learning rate
    print(f"\n🔄 Recompiling model with LR={Config.FINE_TUNE_LR:.2e}...")
    model.compile(
        optimizer=Adam(learning_rate=Config.FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print parameter counts
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print(f"✅ Model recompiled")
    print(f"   → Trainable params: {trainable_params:,} (increased!)")
    print(f"   → Non-trainable params: {non_trainable_params:,}")
    
    # Setup callbacks for fine-tuning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    finetuned_model_path = os.path.join(Config.MODEL_DIR, f'trashformer_finetuned_{timestamp}.keras')
    
    checkpoint_ft = ModelCheckpoint(
        finetuned_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )
    
    early_stop_ft = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )
    
    training_logger_ft = TrainingLogger()
    
    print(f"\n⚙️  Fine-tuning for {Config.FINE_TUNE_EPOCHS} epochs...")
    print(f"💾 Best fine-tuned model will be saved to: {finetuned_model_path}")
    
    # Fine-tune
    try:
        history_fine = model.fit(
            train_generator,
            epochs=Config.FINE_TUNE_EPOCHS,
            validation_data=val_generator,
            callbacks=[checkpoint_ft, early_stop_ft, training_logger_ft],
            verbose=2
        )
        
        # Combine histories
        combined_history = {
            key: initial_history.history[key] + history_fine.history[key]
            for key in initial_history.history.keys()
        }
        
        print("\n" + "="*70)
        print("✅ FINE-TUNING COMPLETED")
        print("="*70)
        print(f"💾 Fine-tuned model saved: {finetuned_model_path}")
        print(f"\n📊 Fine-tuning Improvement:")
        print(f"   Before: {max(initial_history.history['val_accuracy']):.4f}")
        print(f"   After:  {max(history_fine.history['val_accuracy']):.4f}")
        improvement = max(history_fine.history['val_accuracy']) - max(initial_history.history['val_accuracy'])
        if improvement > 0:
            print(f"   Gain:   +{improvement:.4f} ({improvement*100:.2f}%) 🎉")
        else:
            print(f"   Change: {improvement:.4f} ({improvement*100:.2f}%)")
        print("="*70 + "\n")
        
        return combined_history, finetuned_model_path
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Fine-tuning interrupted by user!")
        print(f"💾 Best fine-tuned model saved at: {finetuned_model_path}")
        return initial_history.history, finetuned_model_path


# MAIN TRAINING FUNCTION

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("🗑️  TRASHFORMER - Waste Classification Model Training")
    print("="*70)
    
    # Create data generators
    global val_generator  # Needed for the callback
    train_generator, val_generator = create_data_generators()
    
    # Build model
    model, base_model = build_model(train_generator.num_classes)
    
    # Setup callbacks
    callbacks, model_path = setup_callbacks()
    
    # Train model
    try:
        print("\n" + "="*70)
        print("📚 PHASE 1: INITIAL TRAINING (Transfer Learning)")
        print("="*70)
        
        history = model.fit(
            train_generator,
            epochs=Config.EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=2
        )
        
        print("\n✅ Initial training completed!")
        print(f"   Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        
        # Fine-tuning phase
        finetuned_model_path = None
        if Config.ENABLE_FINE_TUNING:
            combined_history, finetuned_model_path = fine_tune_model(
                model, base_model, train_generator, val_generator, history
            )
        else:
            combined_history = history.history
        
        # Save final model
        final_model_path = os.path.join(Config.MODEL_DIR, 'trashformer_final.keras')
        model.save(final_model_path)
        
        # Save training history (including fine-tuning if applicable)
        history_path = '../training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            if isinstance(combined_history, dict):
                history_dict = {
                    key: [float(val) for val in values] 
                    for key, values in combined_history.items()
                }
            else:
                history_dict = {
                    key: [float(val) for val in values] 
                    for key, values in combined_history.history.items()
                }
            json.dump(history_dict, f, indent=2)
        
        print("\n" + "="*70)
        print("💾 MODELS SAVED:")
        print(f"   → Best initial model: {model_path}")
        if finetuned_model_path:
            print(f"   → Best fine-tuned model: {finetuned_model_path} ⭐")
        print(f"   → Final model: {final_model_path}")
        print(f"   → Training history: {history_path}")
        print("="*70)
        
        # Print final statistics
        print("\n📊 FINAL TRAINING STATISTICS:")
        if isinstance(combined_history, dict):
            final_val_acc = max(combined_history['val_accuracy'])
            final_val_loss = min(combined_history['val_loss'])
            total_epochs = len(combined_history['loss'])
        else:
            final_val_acc = max(combined_history.history['val_accuracy'])
            final_val_loss = min(combined_history.history['val_loss'])
            total_epochs = len(combined_history.history['loss'])
        
        print(f"   → Best Validation Accuracy: {final_val_acc:.4f}")
        print(f"   → Best Validation Loss: {final_val_loss:.4f}")
        print(f"   → Total Epochs Completed: {total_epochs}")
        
        if Config.ENABLE_FINE_TUNING:
            initial_best = max(history.history['val_accuracy'])
            improvement = final_val_acc - initial_best
            print(f"\n🎯 Fine-tuning Impact:")
            print(f"   → Before fine-tuning: {initial_best:.4f}")
            print(f"   → After fine-tuning:  {final_val_acc:.4f}")
            if improvement > 0:
                print(f"   → Improvement: +{improvement:.4f} ({improvement*100:.2f}%) 🎉")
            else:
                print(f"   → Change: {improvement:.4f} ({improvement*100:.2f}%)")
        
        print("="*70)
        
        print("\n💡 To visualize training results, run:")
        print("   python visualize_training.py")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print(f"💾 Best model saved at: {model_path}")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()
