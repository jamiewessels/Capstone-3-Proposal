import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize

def preprocess_img(img):
    loaded = load_img(img_filepath)
    array_img = img_to_array(loaded)
    array_img = resize(array_img,[299,299])
    array_img = img_to_array(array_img)
    return array_img


def predict_img(img_filepath, model_filepath, save_name=None, classes = np.array(['Am', 'C', 'Dm', 'F', 'G'])):
    model = load_model(model_filepath)
    arr = preprocess_img(img_filepath)/255
    # processed = preprocess_input(arr)
    pred = classes[np.argmax(model.predict(arr.reshape(-1, 299, 299, 3)), axis = 1)][0]
    
    fig = plt.gcf()
    fig, ax = plt.subplots()
    ax.imshow(arr)
    ax.text(0.6, 0.7, f'{pred}', size=30, rotation='horizontal',
         ha="center", va="center",
         bbox=dict(boxstyle="square",
                   ec=(0., 0.5, 0.5),
                   fc=(1, 1, 1)
                   )
         )
    fig.savefig('images/predictions/' + save_name)
    fig.show()
    
    return pred

if __name__ == '__main__':
    model_filepath = "CovNet_logs/best_model_5chords.hdf5"
    img_filepath = "images/to_predict/google1.png"
    save_name = 'googleimg4.png'

    predict_img(img_filepath,model_filepath,  save_name)
