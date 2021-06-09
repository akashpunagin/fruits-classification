# import tensorflow as tf
import keras
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

print("MODULES IMPORTED")

# Config
custom_model_path = "models/my_custom_model.h5";
custom_testing_images_path = "testing_images";
classes = ['fresh apples', 'fresh banana', 'fresh oranges', 'rotten apple', 'rotten banana', 'rotten orange']

model = keras.models.load_model(custom_model_path)

print("MODEL SUMMARY\n")
# print(model.summary())
print("Layers in model: ", len(model.layers))

def get_pre_processed_img(image_test):
    img = Image.fromarray(image_test, 'RGB')
    img = img.resize((150, 150))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    return img

def start_video_capture():
    vid = cv2.VideoCapture(0)

    while(True):
        ret, frame = vid.read()

        img = get_pre_processed_img(frame)
        prediction = model.predict(img)
        confidence_score = np.amax(prediction)
        if (confidence_score > 0.90):
            predicted_class = classes[np.argmax(prediction)]
        else:
            predicted_class = "No Fruit Detected"

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = cv2.putText(frame, predicted_class , org, font,
                           fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

def predict_custom_images():
    print("Predicting...")

    file_names = []
    predictions = []

    for file_name in os.listdir(custom_testing_images_path):
        im_path =os.path.join(custom_testing_images_path, file_name)
        image_test = cv2.imread(im_path)
        img = get_pre_processed_img(image_test)

        prediction = model.predict(img)
        confidence_score = np.amax(prediction)
        if (confidence_score > 0.90):
            predicted_class = classes[np.argmax(prediction)]
        else:
            predicted_class = "No Fruit Detected"

        file_names.append(file_name)
        predictions.append(predicted_class)

    data = {"File Name": file_names, "Prediction": predictions}
    df = pd.DataFrame(data)

    print(df.to_string(index=False))


def main():
    predict_custom_images()
    # start_video_capture()

if __name__ == '__main__':
    main()
