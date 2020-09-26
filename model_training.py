import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths   # imutils is for image processing such as transition, rotation

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset = r'E:\Python\Mask_Detection'
imagePaths = list(paths.list_images(dataset))

data = []
labels = []

for i in imagePaths:
    label = i.split(os.path.sep)[-2]   # as it second last from right
    labels.append(label)
    image = load_img(i, target_size=(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)


data = np.array(data, dtype = 'float32')
labels = np.array(labels)

data
labels


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

labels

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size = 0.20, random_state = 10, stratify =labels)

# use image generator to generate more number of images

aug = ImageDataGenerator(rotation_range = 20,zoom_range= 0.15,width_shift_range=0.20,height_shift_range=0.20,shear_range=0.15,horizontal_flip=True,vertical_flip=True)

baseModel = MobileNetV2(weights = 'imagenet',include_top = False, input_tensor = Input(shape=(224,224,3)))
baseModel.summary()


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name = 'Flatten')(headModel)
headModel = Dense(128,activation ='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = 'softmax')(headModel)


model = Model(inputs = baseModel.input, outputs = headModel)
for layer in baseModel.layers:
    layer.trainable= False

model.summary()

# learning rate of the model

learning_rate = 0.001
Epochs = 20
BS = 8

opt = Adam(lr=learning_rate,decay = learning_rate/Epochs)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

model.fit(
    aug.flow( x_train, y_train,batch_size = BS),
    steps_per_epoch = len(x_train)//BS,
    validation_data = (x_test,y_test),
    validation_steps = len(x_test)//BS,
    epochs = Epochs
)

model.save(r'E:\Python\Mask_Detection\mobilenet_v2.model')

predict = model.predict(x_test,batch_size = BS)
predict = np.argmax(predict,axis=1)
print(classification_report(y_test.argmax(axis=1),predict,target_names = lb.classes_))




