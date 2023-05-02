import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from IPython.display import display
from tensorflow import keras
from keras import layers
from keras import models
import numpy as np
import os
from keras.utils import to_categorical

lookup = {'palm': 0,'l': 1,'fist': 2,'fist_moved': 3,'thumb': 4,'index': 5,'ok': 6,'palm_moved': 7,'c': 8,'down': 9}

class Application:
    #Set up all assets
    def __init__(self, window):
        self.window = window
        self.window.title("Hand Gesture Recognition")
        self.window.configure(bg="#404040")
        #Frame
        self.video_frame = tk.Frame(self.window, bg="#404040")
        self.video_frame.pack(side=tk.TOP)
        #Title Label
        self.title_label = tk.Label(self.video_frame, text="Super Sick Hand Gesture Recognition", bg="#404040", fg="white", font=("Comic Sans MS", 20))
        self.title_label.pack(side=tk.TOP, pady=10)
        #Snap Button
        self.snap_button = tk.Button(self.video_frame, text="Snap", command=self.take_snapshot, bg="#606060", fg="black")
        self.snap_button.pack(side=tk.TOP, pady=10)
        #Video Frame
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(side=tk.TOP, pady=10)
        #Predicted Gesture Label
        self.gesture_label = tk.Label(self.window, text="Predicted gesture:", bg="#404040", fg="white", font=("Comic Sans MS", 16))
        self.gesture_label.pack(side=tk.BOTTOM, pady=0)
        #Video capture
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.update_video()


    #Video updates every 15 ms
    def update_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = image.resize((640, 480))
            photo = ImageTk.PhotoImage(image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        self.window.after(15, self.update_video)


    #Take a frame  and make a prediction
    def take_snapshot(self):
        ret, frame = self.video_capture.read()
        if ret:
            #Grayscale + save image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = image.convert('L')
            image.save("snapshot.png")

            #Resize image
            image = image.resize((320, 120))
            image = np.array(image)
            image = image.astype('float32') / 255
            image = np.expand_dims(image, axis=0)

            # Load the saved model
            model = tf.keras.models.load_model("gesture_model.h5")

            # Make a prediction using the model
            prediction = model.predict(image)

            print(prediction)

            # Get the index of the predicted gesture
            predicted_gesture_index = np.argmax(prediction[0])

            # Look up the predicted gesture name in the lookup dictionary
            predicted_gesture_name = list(lookup.keys())[list(lookup.values()).index(predicted_gesture_index)]
            

            # Display the predicted gesture name in the GUI label
            self.gesture_label.configure(text=f"Predicted gesture: {predicted_gesture_name}")



# Create the application window
window = tk.Tk()

# Set the window size and position
window.geometry("640x600+100+100")

# Start the application
app = Application(window)

# Run the main window loop
window.mainloop()
