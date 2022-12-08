# 라이브러리 임포트
import os # file path
import random # random
import cv2 # image
import pandas as pd # dataframe
import numpy as np # numpy
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Convolution2D,Flatten,Dense,MaxPooling2D,Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/Administrator/PycharmProjects/pythonProject1/new_df_1.csv')
df['path'] = df['path'].str.replace('/home/pi/IoT_class_Data_collec/AIoT/','')

def list_dir(path):
  filenames = os.listdir(path)
  filenames.sort()
  return filenames

def resize_img(imgpath,resolution):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img, resolution)
    return np.array(img_resize)


def load_image_pixels(imagepath, resolution):
    img         = Image.open(imagepath)
    img_resized = img.resize(resolution)
    return np.array(img_resized)


images = []  # 이미지 넣는데
path = 'C:/Users/Administrator/PycharmProjects/pythonProject1/pra_img'
target_names = list_dir(path)
print(target_names)
for fname in df['path']:
    image_path = os.path.join('C:/Users/Administrator/PycharmProjects/pythonProject1/pra_img', fname)
    pixels = load_image_pixels(image_path, [150, 150])
    images.append(pixels)

xs = np.asarray(images, dtype=np.float32)
ys = df['angle']
ys = np.asarray(ys)

xs_norm = (xs-xs.min() ) / (xs.max()-xs.min())
train_X, test_X,train_y, test_y = train_test_split(xs_norm,ys,
                                               test_size=0.3, random_state=10)


from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from tensorflow.keras.models import Model


_, img_height, img_width , img_channel = train_X.shape
inputs = Input(shape = (img_height, img_width, img_channel))
x = Conv2D(kernel_size = (3,3) ,filters = 64,padding = 'same', activation = 'elu')(inputs)
x = Conv2D(kernel_size = (3,3), filters=128 , padding ='same', activation = 'elu')(x)
x = MaxPooling2D((2,2),strides =(2,2))(x)
x = Dropout(rate=0.5)(x)
x = Conv2D(kernel_size=(3,3),filters = 128, padding = 'same', activation='elu')(x)
x = Conv2D(kernel_size=(3,3), filters = 256, padding = 'valid', activation='elu')(x)
x = MaxPooling2D((2,2),strides = (2,2))(x)
x = Conv2D(kernel_size=(3,3),filters = 128, padding = 'same', activation='elu')(x)
x = Conv2D(kernel_size=(3,3), filters = 256, padding = 'valid', activation='elu')(x)
x = MaxPooling2D((2,2),strides = (2,2))(x)
x = Conv2D(kernel_size=(3,3),filters = 128, padding = 'same', activation='elu')(x)
x = Conv2D(kernel_size=(3,3), filters = 256, padding = 'valid', activation='elu')(x)
x = MaxPooling2D((2,2),strides = (2,2))(x)
x = Conv2D(kernel_size=(3,3),filters = 128, padding = 'same', activation='elu')(x)
x = Conv2D(kernel_size=(3,3), filters = 256, padding = 'valid', activation='elu')(x)
x = MaxPooling2D((2,2),strides = (2,2))(x)
x = Dropout(rate=0.3)(x)
x = Flatten()(x)
# x = Dense(units=512, activation = 'relu')(x)
# x = Dropout(rate = 0.3)(x)
x = Dense(units=256, activation = 'elu')(x)
x = Dense(units=128, activation = 'elu')(x)
x = Dense(units=64, activation = 'elu')(x)
x = Dropout(rate = 0.3)(x)
output = Dense(units=1, activation = 'elu')(x)
model = Model(inputs=inputs, outputs=output)

model.compile(Adam(lr=0.00007),loss='mse')

history = model.fit(train_X, train_y,
                      epochs = 60,
                      validation_split = 0.3,
                      batch_size = 8,
                   )


# model.evaluate(test_X, test_y)
#
# import matplotlib.pyplot as plt
#
# img = xs[1]
# img_image = Image.fromarray(np.uint8(img))
# plt.imshow(img_image)
#
# plt.figure(figsize = (12,4))
#
# plt.plot(history.history['loss'],    'b-o', label = 'loss')
# plt.plot(history.history['val_loss'],'r--o',label = 'val_loss')
# plt.xlabel('Epoch')
#
# plt.grid()
# plt.legend()
# plt.show()

pred_y = model.predict(test_X)
#eval_list = [a for i in range(1,10) a = random.randint(1,123)]
import random
eval_list = []
for i in range(1,10):
    b = random.randint(0,20)
    eval_list.append(b)
for test_index in eval_list:
    print("pred_y_{} -> {}".format(test_index, pred_y[test_index]))
    print("test_y_{} -> {}".format(test_index, test_y[test_index]))
    print("pred_y_Prob : {}".format(np.round(pred_y[test_index],3)))
    img = test_X[test_index]
    plt.imshow(img)
    plt.axis('off')
    plt.show()

model.save('my_model_4.h5')