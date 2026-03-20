import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

def load_assets():
    model = tf.keras.models.load_model("./model/ann_model1.h5")

    with open("./utils/scaling.pkl", 'rb') as file:
        scaling = pickle.load(file)

    return model, scaling

model, scaler = load_assets()

canvas_width = 300
canvas_height = 300

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="black")
        self.canvas.pack()

        self.button_predict = tk.Button(root, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.button_clear = tk.Button(root, text="Clear", command=self.clear)
        self.button_clear.pack()

        self.label = tk.Label(root, text="Draw a digit")
        self.label.pack()

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.last_x, self.last_y = None, None

        self.image = Image.new("L", (canvas_width, canvas_height), "black")
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=6, fill="white")
            self.draw_image.line([self.last_x, self.last_y, event.x, event.y], fill="white", width=6)

        self.last_x = event.x
        self.last_y = event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.draw_image.rectangle([0, 0, canvas_width, canvas_height], fill="black")
        self.last_x, self.last_y = None, None

    def preprocess_image(self, img):
        """
        Replicates standard MNIST preprocessing:
        1. Resizes while preserving aspect ratio.
        2. Centers the digit based on its Center of Mass inside a 28x28 canvas.
        """
        # Thicken the lines slightly before resizing
        img = img.filter(ImageFilter.MaxFilter(3)) 
        
        # Preserve Aspect Ratio and scale to max 20x20
        w, h = img.size
        if w > h:
            new_w = 20
            new_h = int(20 * (h / w))
        else:
            new_h = 20
            new_w = int(20 * (w / h))
            
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        img_array = np.array(img_resized)

        # Calculate Center of Mass using Numpy
        total_mass = np.sum(img_array)
        if total_mass > 0:
            y_indices, x_indices = np.indices(img_array.shape)
            cy = np.sum(y_indices * img_array) / total_mass
            cx = np.sum(x_indices * img_array) / total_mass
        else:
            cy, cx = new_h / 2, new_w / 2

        # Create a clean 28x28 canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)

        # Calculate where to place the image so the center of mass lands on (14, 14)
        y_offset = int(np.round(14.0 - cy))
        x_offset = int(np.round(14.0 - cx))

        # Safely paste the image array into the canvas array
        y1, y2 = max(0, y_offset), min(28, y_offset + new_h)
        x1, x2 = max(0, x_offset), min(28, x_offset + new_w)
        
        sy1, sy2 = max(0, -y_offset), max(0, -y_offset) + (y2 - y1)
        sx1, sx2 = max(0, -x_offset), max(0, -x_offset) + (x2 - x1)

        canvas[y1:y2, x1:x2] = img_array[sy1:sy2, sx1:sx2]

        return Image.fromarray(canvas)

    def predict(self):
        bbox = self.image.getbbox()

        if bbox is None:
            self.label.config(text="Draw something first!")
            return

        # Crop to drawing
        img = self.image.crop(bbox)

        # Apply MNIST-style centering and aspect ratio constraints
        img = self.preprocess_image(img)
        img = img.filter(ImageFilter.SHARPEN)

        # Prepare for model
        img_array = np.array(img)
        
        # --- Save a copy BEFORE scaling so Matplotlib displays it correctly ---
        img_for_display = img_array.copy()
        
        img_flat = img_array.reshape(1, 784)
        
        # Apply the scaler (this is where pixel values get shifted for the ANN)
        img_scaled = scaler.transform(img_flat)

        # Predict
        prediction = model.predict(img_scaled)
        digit = np.argmax(prediction)

        self.label.config(text=f"Prediction: {digit}")
        print(f"Prediction Probabilities: {prediction}")

        # Plot the clean, unscaled visualization
        plt.imshow(img_for_display, cmap="gray", vmin=0, vmax=255)
        plt.title(f"Model Input (Predicted: {digit})")
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()