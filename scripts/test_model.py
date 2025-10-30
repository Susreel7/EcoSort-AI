#!/usr/bin/env python3
"""
Test the trained Trashformer model on individual images
"""

import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import glob

# Class names (must match training order)
CLASS_NAMES = ['cardboard', 'e-waste', 'glass', 'medical', 'metal', 'paper', 'plastic']

def load_model(model_path=None):
    """Load the trained model"""
    if model_path is None:
        # Look for model files in multiple locations
        model_patterns = [
            '../models/trashformer_finetuned_*.keras',
            '../models/trashformer_best_*.keras',
            '../models/trashformer_final*.keras',
            '../models/*.keras',
            'models/*.keras'
        ]
        
        model_files = []
        for pattern in model_patterns:
            model_files.extend(glob.glob(pattern))
        
        if not model_files:
            print("❌ No trained models found in models/ directory")
            print("📁 Checked paths:")
            for pattern in model_patterns:
                print(f"   - {pattern}")
            print("💡 Train a model first: python train_trashformer.py")
            return None
        model_path = max(model_files, key=os.path.getctime)
    
    print(f"📦 Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for prediction"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed (handles PNG with alpha, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

def predict_image(model, image_path, show_all_probs=False):
    """Make prediction on a single image"""
    if not show_all_probs:
        print(f"\n{'='*70}")
        print(f"🖼️  Analyzing: {os.path.basename(image_path)}")
        print(f"{'='*70}")
    
    # Preprocess image
    img_array = preprocess_image(image_path)
    if img_array is None:
        return None, 0.0
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]
    
    # Get top prediction
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = probabilities[predicted_class_idx] * 100
    
    # Display result
    if not show_all_probs:
        print(f"\n🎯 PREDICTION: {predicted_class.upper()}")
        print(f"📊 Confidence: {confidence:.2f}%")
    
    # Show all probabilities if requested
    if show_all_probs:
        print(f"🖼️  {os.path.basename(image_path)}")
        print(f"🎯 PREDICTION: {predicted_class.upper()} ({confidence:.2f}%)")
        print(f"\n📈 All Class Probabilities:")
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            class_name = CLASS_NAMES[idx]
            prob = probabilities[idx] * 100
            bar = '█' * int(prob / 2)
            print(f"   {class_name:12s} {prob:6.2f}% {bar}")
    
    if not show_all_probs:
        print(f"{'='*70}\n")
    
    return predicted_class, confidence

def evaluate_on_validation_set(model):
    """Evaluate model on the entire validation set"""
    print("\n" + "="*70)
    print("📊 EVALUATING ON VALIDATION SET")
    print("="*70)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Find validation directory - try multiple paths
    val_paths = ['../waste_data_split/val', 'waste_data_split/val', './waste_data_split/val']
    val_dir = None
    
    for path in val_paths:
        if os.path.exists(path):
            val_dir = path
            break
    
    if val_dir is None:
        print("❌ Validation directory not found!")
        print(f"📁 Tried: {val_paths}")
        return None, None
    
    print(f"📁 Using validation directory: {val_dir}")
    
    # Load validation data
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate
    print(f"📊 Found {val_generator.samples} validation images")
    print("\n⏳ Evaluating... (this may take a moment)\n")
    results = model.evaluate(val_generator, verbose=1)
    loss = results[0]
    accuracy = results[1]
    
    print(f"\n✅ VALIDATION RESULTS:")
    print(f"   Loss: {loss:.4f}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*70 + "\n")
    
    return accuracy, loss

def test_random_samples(model, num_samples=5):
    """Test on random images from validation set"""
    print("\n" + "="*70)
    print(f"🎲 TESTING ON {num_samples} RANDOM SAMPLES")
    print("="*70)
    
    # Get random images from validation set - try multiple paths
    val_paths = ['../waste_data_split/val', 'waste_data_split/val', './waste_data_split/val']
    val_dir = None
    
    for path in val_paths:
        if os.path.exists(path):
            val_dir = path
            break
    
    if val_dir is None:
        print("❌ Validation directory not found!")
        print(f"📁 Tried: {val_paths}")
        return
    
    print(f"📁 Using validation directory: {val_dir}\n")
    
    all_images = []
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(val_dir, class_name)
        if os.path.exists(class_dir):
            images = glob.glob(os.path.join(class_dir, '*.jpg'))
            all_images.extend([(img, class_name) for img in images])
    
    if not all_images:
        print("❌ No validation images found!")
        return
    
    print(f"📊 Found {len(all_images)} total validation images")
    
    # Sample random images
    if len(all_images) > num_samples:
        import random
        samples = random.sample(all_images, num_samples)
    else:
        samples = all_images[:num_samples]
    
    # Test each sample
    correct = 0
    for i, (image_path, true_class) in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}]", end=" ")
        predicted_class, confidence = predict_image(model, image_path, show_all_probs=False)
        
        if predicted_class is None:
            print(f"⚠️  SKIPPED (Error processing image)\n")
            continue
        
        if predicted_class == true_class:
            print(f"✅ CORRECT! (True: {true_class}, Predicted: {predicted_class}, Confidence: {confidence:.1f}%)")
            correct += 1
        else:
            print(f"❌ WRONG! (True: {true_class}, Predicted: {predicted_class}, Confidence: {confidence:.1f}%)")
    
    print("\n" + "="*70)
    print(f"📊 Random Sample Accuracy: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
    print("="*70 + "\n")

def main():
    """Main function"""
    import sys
    
    print("\n" + "="*70)
    print("🗑️  TRASHFORMER - Model Testing")
    print("="*70)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Check for command line argument
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\n🔍 What would you like to do?\n")
        print("1. Test a specific image")
        print("2. Evaluate on full validation set")
        print("3. Test on random samples (10 samples)")
        print("4. All of the above")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Exiting...")
            return
    
    if choice == '1':
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            predict_image(model, image_path, show_all_probs=True)
        else:
            print(f"❌ Image not found: {image_path}")
    
    elif choice == '2':
        evaluate_on_validation_set(model)
    
    elif choice == '3':
        try:
            num_samples = input("How many random samples? (default: 10): ").strip()
            num_samples = int(num_samples) if num_samples else 10
        except (EOFError, KeyboardInterrupt, ValueError):
            num_samples = 10
        test_random_samples(model, num_samples)
    
    elif choice == '4':
        # Full evaluation
        print("\n🔄 Running full evaluation...")
        evaluate_on_validation_set(model)
        test_random_samples(model, num_samples=10)
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()

