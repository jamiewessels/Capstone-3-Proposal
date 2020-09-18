import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
import PIL




def preprocess_img(img_filepath, rot = -90):
    loaded = load_img(img_filepath)
    if loaded.size[0]< loaded.size[1]:
        loaded = loaded.rotate(rot)
        img = loaded.resize((299,299))
    else:
        img = loaded.resize((299,299))
    img = img_to_array(img)/255
    # img = preprocess_input(img).astype('float32')
    return img


def predict_img(img_filepath, model_filepath, save_name=None, classes = np.array(['Am', 'C', 'Dm', 'F', 'G'])):
    model = load_model(model_filepath)
    arr = preprocess_img(img_filepath)

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
    model_filepath = "CovNet_logs/best_model_5chords_.hdf5"
    img_filepath = "images/to_predict/google5.png"
    save_name = 'googleimg5.png'

    predict_img(img_filepath,model_filepath,  save_name)
