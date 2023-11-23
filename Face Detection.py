from keras.models import load_model  # TensorFlow is required for Keras to w
import cv2  # Install opencv-python
import numpy as np
#from google.colab.patches import cv2_imshow

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)


# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
   
# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    img1=image.copy()

    # Resize the raw image into (224-height,224-width) pixels
    if ret:
      image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    #image=cv2.imread(image)
    

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    if index == 0:
        text="human face detected"
    else:
        text="searching...."
    img1 = cv2.putText(img1, text, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    
    cv2.imshow('frame',img1)

    # Listen to the keyboard for presses.
    #keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

camera.release()
cv2.destroyAllWindows()

