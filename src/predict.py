import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import PIL


def preprocess_img(img_filepath, rot, crop):
    loaded = load_img(img_filepath)
    #if crop, crop to square first so image not too distorted from reality
    if crop and rot:
        loaded = loaded.rotate(rot)
        w, h = loaded.size
        img = loaded.crop((w//4, w//4, 3*w//4, 3*w//4))
        img = img.resize((299,299))
    elif rot: 
        loaded = loaded.rotate(rot)
        img = loaded.resize((299,299))
    else:
        img = loaded.resize((299,299))
    arr = img_to_array(img)
    img_for_pic = arr/255 #color scaling for picture of img
    img_for_model = preprocess_input(arr) #color scaling for model
    return img_for_pic, img_for_model


def predict_img(img_filepath, model_filepath, save_name=None,rot = -90, crop = False, classes = np.array(['Am', 'C', 'Dm', 'F', 'G'])):
    model = load_model(model_filepath)
    img_for_pic, img_for_model = preprocess_img(img_filepath, rot, crop)

    pred= classes[np.argmax(model.predict(img_for_model.reshape(-1, 299, 299, 3)), axis = 1)][0]
    
    fig = plt.gcf()
    fig, ax = plt.subplots()
    ax.imshow(img_for_pic)
    ax.text(50,50, f'{pred}', size=30, rotation='horizontal',
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
    img_filepath = "images/to_predict/Am/google2.png"
    save_name = 'google2.png'

    predict_img(img_filepath,model_filepath,  save_name, rot = 0, crop = False)