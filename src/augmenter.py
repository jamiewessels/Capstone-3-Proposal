from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='constant')

cur_dir = 'data/kate_pictures/train/G'

for filename in os.listdir(cur_dir):
        loc = cur_dir + '/' + str(filename)
        print(filename)
        img = load_img(loc)  
        x = img_to_array(img)  
        x = x.reshape((1,) + x.shape)  



        i = 5000
        for batch in datagen.flow(x, batch_size=1, save_to_dir='data/train/augmented_2/G', save_prefix='G', save_format='jpg'):
                i += 1
                if i > 5020:
                        break 