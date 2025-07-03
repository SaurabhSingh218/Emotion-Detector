import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import tensorflow as tf

# ---------------------------
# Config
MODEL_FILE = "emotion_model.h5"  # or "emotion_model.keras"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ---------------------------
# Load the model with error handling
def FacialExpressionModel(model_file):
    if not os.path.exists(model_file):
        messagebox.showerror("Error", f"Model file not found: {model_file}")
        exit()
    
    try:
        # Handle the batch_shape error by monkey-patching the problematic parameter
        print("Attempting to load model with compatibility fixes...")
        
        # Import required modules for patching
        import tensorflow.keras.layers as layers
        import tensorflow.python.keras.layers as python_layers
        
        # Store original methods
        original_input_layer_init = None
        if hasattr(layers, 'InputLayer'):
            original_input_layer_init = layers.InputLayer.__init__
        
        def patched_input_layer_init(self, *args, **kwargs):
            # Remove problematic batch_shape parameter
            if 'batch_shape' in kwargs:
                print("Removing batch_shape parameter...")
                kwargs.pop('batch_shape')
            return original_input_layer_init(self, *args, **kwargs)
        
        # Apply patch
        if original_input_layer_init:
            layers.InputLayer.__init__ = patched_input_layer_init
            if hasattr(python_layers, 'InputLayer'):
                python_layers.InputLayer.__init__ = patched_input_layer_init
        
        # Try loading the model
        model = load_model(model_file, compile=False)
        
        # Restore original method
        if original_input_layer_init:
            layers.InputLayer.__init__ = original_input_layer_init
            if hasattr(python_layers, 'InputLayer'):
                python_layers.InputLayer.__init__ = original_input_layer_init
        
        print(f"✅ Model loaded successfully: {model_file}")
        return model
        
    except Exception as e:
        print(f"Patching method failed: {e}")
        
        # Restore original method if patching failed
        if 'original_input_layer_init' in locals() and original_input_layer_init:
            layers.InputLayer.__init__ = original_input_layer_init
            if hasattr(python_layers, 'InputLayer'):
                python_layers.InputLayer.__init__ = original_input_layer_init
        
        # Simple fallback - try loading your existing model directly
        try:
            print("Trying simple load_model...")
            import warnings
            warnings.filterwarnings('ignore')
            model = tf.keras.models.load_model(model_file, compile=False)
            print(f"✅ Model loaded with simple method: {model_file}")
            return model
        except Exception as e2:
            print(f"Simple loading failed: {e2}")
            messagebox.showerror("Error", f"Cannot load your existing model due to TensorFlow version incompatibility.\n\nQuick fix: Try running:\npip install tensorflow==2.6.0\n\nOr provide the original training script to re-save the model.")
            exit()

model = FacialExpressionModel(MODEL_FILE)

# ---------------------------
# Load Haar Cascade
if not os.path.exists(CASCADE_FILE):
    messagebox.showerror("Error", f"Cascade file not found: {CASCADE_FILE}\nDownload it from OpenCV repo.")
    exit()

facec = cv2.CascadeClassifier(CASCADE_FILE)

# ---------------------------
# GUI Setup
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def Detect(file_path):
    """Detect emotion from uploaded image"""
    try:
        image = cv2.imread(file_path)
        if image is None:
            label1.configure(foreground="#011638", text="Unable to read image")
            return

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image, 1.3, 5)

        if len(faces) == 0:
            label1.configure(foreground="#011638", text="No face detected")
            return

        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            roi = roi.astype('float32') / 255.0
            
            # Ensure proper shape for prediction
            roi_batch = np.expand_dims(roi, axis=0)
            roi_batch = np.expand_dims(roi_batch, axis=-1)
            
            prediction = model.predict(roi_batch, verbose=0)
            pred_emotion = EMOTIONS_LIST[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            print(f"Predicted: {pred_emotion} ({confidence:.2f}%)")
            label1.configure(foreground="#011638", text=f"{pred_emotion} ({confidence:.1f}%)")
            break

    except Exception as e:
        print(f"Error in detection: {e}")
        label1.configure(foreground="#011638", text="Detection error")

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Emotion",
                      command=lambda: Detect(file_path),
                      padx=10, pady=5,
                      background="#364156", foreground='white',
                      font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )

        if file_path:
            uploaded = Image.open(file_path)
            max_width = int(top.winfo_width() / 1.5)
            max_height = int(top.winfo_height() / 1.5)
            if max_width <= 0: max_width = 400
            if max_height <= 0: max_height = 300
            uploaded.thumbnail((max_width, max_height))
            im = ImageTk.PhotoImage(uploaded)

            sign_image.configure(image=im)
            sign_image.image = im
            label1.configure(text='')
            show_Detect_button(file_path)

    except Exception as e:
        print(f"Upload error: {e}")
        messagebox.showerror("Error", f"Failed to upload: {e}")

# ---------------------------
# GUI Layout
upload = Button(top, text="Upload Image", command=upload_image,
                padx=10, pady=5,
                background="#364156", foreground='white',
                font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)

heading = Label(top, text='Emotion Detector', pady=20,
                font=('arial', 25, 'bold'),
                background='#CDCDCD', foreground="#364156")
heading.pack()

status_label = Label(top, text="Ready. Upload an image to detect emotions.",
                     background='#CDCDCD', font=('arial', 10),
                     foreground='green')
status_label.pack()

print("✅ GUI ready")
print(f"TensorFlow version: {tf.__version__}")
top.mainloop()