import os
from sklearn.metrics import confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

#Assigning the data directory.
dataDir = "Combined Dataset"

#Assigning each class to the classes array.
class_names = ["del","spc","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W",
               "X","Y","Z"]

processed_Data = []

#Preporcessing subclass
def preprocessData():
    for x in class_names:
        #creating a path for each image in the directory with the class array.
        path = os.path.join(dataDir,x)
        #Assinging a class number based on the index of each class in the classes array.
        class_num = class_names.index(x)
        for img in os.listdir(path):
            #Reading each image with the path and then converting image to gray.
            img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            try:
                #Converting the size of the image to 60 by 60
                img_arr2 = cv2.resize(img_arr, dsize=(60,60))

            except Exception as e:
                #In case of failure of converting a specific image the path is printed of the image with the error.
                print(path,img)
            processed_Data.append([img_arr2,class_num])

preprocessData()


images =[]
labels=[]

for img, lab in processed_Data:
    images.append(img)
    labels.append(lab)

#Reshaping the array to old dimensions.
images = np.array(images).reshape(-1,60,60,1)
labels = np.array(labels)

#Data split of:
#Train 60% Test 20% Validation 20%
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=3)


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=3)

#Normalising the data for the neural network.
#All the numbers are in the range of 256 as they are colors so dividing by 255 will give a range
#from 0 to 1 making it easier for the neural network to interpret.
X_train = X_train/255.0
X_test = X_test/255.0

print(X_train[1].shape)

# Setting up a sequential neural network with 7 layers.
# model = tf.keras.Sequential([
#         #A convolutional layer with 16 filters and a kernel with a dimension of 3x3.
#         tf.keras.layers.Conv2D(16, (3,3), activation ='relu', input_shape=X_train[1].shape),
#         #Same layer as first but will learn unique features as the filters is randomised.
#         tf.keras.layers.Conv2D(16, (3,3), activation ='relu'),
#         #Maxpooling is used to reduce spatial dimensions of the output volume.
#         #creates a feature map based of previous map containing important features.
#         tf.keras.layers.MaxPool2D((2,2)),
#         tf.keras.layers.Dense(128, activation="relu"),
#         #flattening the data to a 1 dimensional array for next layer
#         tf.keras.layers.Flatten(),
#         #Dense layer with 50 neurons.
#         tf.keras.layers.Dense(50, activation='relu'),
#         #An output dense layer contain 28 neurons 26 for alphabet and 2 for space and delete gestures.
#         tf.keras.layers.Dense(28, activation='softmax')
# ])
#
# model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
# #Early stop monitor to prevent over fitting by measuring validation loss once validation loss increases
# #the neural network will stop training.
#
#
# monitor = EarlyStopping(monitor = "val_loss",min_delta=1e-3,patience=5,verbose=1,mode="auto"
#                         ,restore_best_weights=True)
#
# model.fit(X_train,Y_train,validation_data=(X_val,Y_val),callbacks=[monitor],verbose=2,epochs=1000)

model = keras.models.load_model("ASLModelCombined5.h5")

model.summary()
prediction = model.predict(X_val)

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'black','size':15}
predTrue = 0
predFalse= 0


for i in range(200):
    plt.grid(False)
    plt.title("Prediction Number:"+str(i+1),font1)
    plt.imshow(X_val[i],cmap="gray")
    plt.ylabel("Actual: "+class_names[Y_val[i]],font2)
    plt.xlabel("Prediction: "+ class_names[np.argmax(prediction[i])],font2)
    if class_names[Y_val[i]] == class_names[np.argmax(prediction[i])]:
        predTrue+=1
    else:
        predFalse+=1
    plt.show()

print("Predicted Correct: "+str(predTrue))
print("Predicted Incorrect: "+str(predFalse))

# model.save("ASLmodelCombined5.h5")