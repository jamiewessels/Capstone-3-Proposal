import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize

def preprocess_img(img_filepath):
    loaded = load_img(img_filepath)
    array_img = img_to_array(loaded)
    array_img = resize(array_img,[299,299])
    array_img = img_to_array(array_img)
    return array_img


def predict_img(model_filepath, img_filepath, classes = np.array(['Am', 'C', 'Dm', 'F', 'G'])):
    model = load_model(model_filepath)
    arr = preprocess_img(img_filepath)/255
    # processed = preprocess_input(arr)
    pred = classes[np.argmax(model.predict(arr.reshape(-1, 299, 299, 3)), axis = 1)][0]

    fig = plt.gcf()
    fig, ax = plt.subplots()
    ax.imshow(arr)
    ax.text(0.6, 0.7, f'{pred}', size=50, rotation=30.,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
    fig.show()

    return pred

if __name__ == '__main__':
    model_filepath = "CovNet_logs/Checkpoint5c-4.hdf5"
    img_filepath = "src/trial.jpeg"

    print(predict_img(model_filepath, img_filepath))
