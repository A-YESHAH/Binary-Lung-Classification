# api/heatmap.py - PROPER Grad-CAM IMPLEMENTATION
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from PIL import Image
import io
import base64

def get_gradcam_heatmap(model, img_array, layer_name=None):
    """
    Generate real Grad-CAM heatmap using model gradients
    """
    try:
        # If no layer specified, try to find the last convolutional layer
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name or 'activation' in layer.name:
                    layer_name = layer.name
                    print(f"📌 Using layer: {layer_name}")
                    break
        
        if layer_name is None:
            raise ValueError("No convolutional layer found in model")
        
        # Get the target layer
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Record operations for gradient computation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # For binary classification (pneumonia vs normal)
            if predictions.shape[1] == 1:
                loss = predictions[:, 0]  # Binary classification
            else:
                # If multi-class, use class with highest probability
                class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the convolutional outputs with pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"❌ Grad-CAM failed: {e}")
        # Fallback to class activation mapping
        return get_cam_heatmap(model, img_array)

def get_cam_heatmap(model, img_array):
    """
    Fallback: Class Activation Mapping (simpler than Grad-CAM)
    """
    try:
        # Get the last convolutional layer
        conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                conv_layer = layer
                break
        
        if conv_layer is None:
            raise ValueError("No convolutional layer found")
        
        # Create model that outputs conv layer and predictions
        cam_model = Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )
        
        # Get convolutional features and predictions
        conv_outputs, predictions = cam_model.predict(img_array, verbose=0)
        conv_outputs = conv_outputs[0]
        
        # Get weights of the last layer (assuming Dense layer after conv)
        last_layer = model.layers[-1]
        if hasattr(last_layer, 'get_weights'):
            weights = last_layer.get_weights()[0]
            
            # For binary classification
            if weights.shape[1] == 1:
                class_weights = weights[:, 0]
            else:
                # Find class with highest probability
                class_idx = np.argmax(predictions[0])
                class_weights = weights[:, class_idx]
            
            # Generate CAM
            heatmap = np.zeros(conv_outputs.shape[:2])
            for i, w in enumerate(class_weights):
                heatmap += w * conv_outputs[:, :, i]
            
            # Normalize
            heatmap = np.maximum(heatmap, 0)
            heatmap_max = np.max(heatmap)
            if heatmap_max > 0:
                heatmap /= heatmap_max
            
            return heatmap
    
    except Exception as e:
        print(f"❌ CAM failed: {e}")
    
    # Ultimate fallback: simple heatmap based on image intensity
    return get_simple_heatmap(img_array[0])

def get_simple_heatmap(img_array):
    """
    Simple intensity-based heatmap (least accurate, but always works)
    """
    # Convert to grayscale
    gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Normalize
    heatmap = cv2.normalize(gray.astype(float), None, 0, 1, cv2.NORM_MINMAX)
    
    # Apply Gaussian blur
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
    return heatmap

def overlay_heatmap_on_image(img_array, heatmap, alpha=0.5):
    """
    Overlay heatmap on original image
    """
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Convert original image
    original_img = (img_array * 255).astype(np.uint8)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    
    # Overlay
    overlayed = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed

def heatmap_to_base64(heatmap_img):
    """
    Convert heatmap image to base64 for frontend
    """
    # Convert BGR to RGB
    heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(heatmap_rgb)
    
    # Save to bytes
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG", optimize=True)
    
    # Convert to base64
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{img_str}"

def analyze_model_layers(model):
    """
    Helper to analyze model layers and find suitable ones for Grad-CAM
    """
    print("🔍 Analyzing model layers...")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name} ({layer.__class__.__name__})")
        if hasattr(layer, 'output_shape'):
            print(f"   Output shape: {layer.output_shape}")
    print()