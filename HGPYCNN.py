import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
directory = r"/content/drive/MyDrive/Database/train/train"
print(os.path.exists(directory))



classes = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
all_arrays=[]
for i in classes:
  path = os.path.join(directory,i)
  class_num = classes.index(i)
  for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img))
    width=100
    height=100
    dim=(width,height)
    reized=cv2.resize(img_array,dim,interpolation=cv2.INTER_AREA)
    img_array = cv2.cvtColor(reized,cv2.COLOR_BGR2RGB)
    all_arrays.append([img_array,class_num])



directory2 = r"/content/drive/MyDrive/Database/test/test"
classes2 = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
all_arrays2=[]
for i in classes2:
  path2=os.path.join(directory2,i)
  class_num2=classes2.index(i)
  for img in os.listdir(path2):
    img_array2=cv2.imread(os.path.join(path2,img))
    width=100
    height=100
    dim=(width,height)
    reized=cv2.resize(img_array2,dim,interpolation=cv2.INTER_AREA)
    img_array2=cv2.cvtColor(reized,cv2.COLOR_BGR2RGB)
    all_arrays2.append([img_array2,class_num2])




hands_array_train=[]
for features,label in all_arrays:
 hands_array_train.append(features)
location= [[0,350,566],[770,1110,1315],[1570,1840,2055]]
hands_gesture=["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
a=0
b=1
c=2
for i,j,k in location:
  plt.subplots(figsize=(7,7))
  plt.subplot(1,3,1)
  plt.imshow(hands_array_train[i])
  plt.title(hands_gesture[a])
  plt.axis("off")
  plt.subplot(1,3,2)
  plt.imshow(hands_array_train[j])
  plt.title(hands_gesture[b])
  plt.axis("off")
  plt.subplot(1,3,3)
  plt.imshow(hands_array_train[k])
  plt.title(hands_gesture[c])
  plt.axis("off")
  a+=3
  b+=3
  c+=3




import random
random.shuffle(all_arrays)
X_train=[]
Y_train=[]
for features,label in all_arrays:
  X_train.append(features)
  Y_train.append(label)
X_train=np.array(X_train)
print(X_train)
print(Y_train)



random.shuffle(all_arrays2)
X_test=[]
Y_test=[]
for features,label in all_arrays2:
  X_test.append(features)
  Y_test.append(label)
X_test=np.array(X_test)




img_size=100
X_train=X_train.reshape(-1,img_size,img_size,3)
X_train=X_train/255

X_test=X_test.reshape(-1,img_size,img_size,3)
X_test=X_test/255
print("shape of X_train= {}",X_train.shape)
print("shape of X_test= {}",X_test.shape)

Y_train=keras.utils.to_categorical(Y_train,num_classes=24)
Y_test=keras.utils.to_categorical(Y_test,num_classes=24)
print(Y_train.shape)



x_train,x_val,y_train,y_val=train_test_split(X_train,
Y_train,test_size=0.3,random_state=40)



from tensorflow.keras import layers,optimizers,Sequential,Input
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
for layer in base_model.layers:
    layer.trainable = False
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(24, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


oAdam=optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=oAdam,loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()
model.save('my_model.h5')




from tensorflow.keras.optimizers import Adam
from keras.models import load_model

model = load_model('my_model.h5')
datagen=ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,height_shift_range=0.2, shear_range=0.15, horizontal_flip=True,
fill_mode="nearest")

 
history = model.fit(x_train, y_train, epochs=5, batch_size=40, validation_data=(x_val, y_val))



 




plt.plot(history.history["val_accuracy"],color="r",label="val_accuracy")
plt.plot(history.history["accuracy"],color="b",label="train_accuracy")
plt.title("accuracy graph")
plt.ylabel("accuracy")
plt.xlabel("number of epoch")
plt.legend()
plt.grid()
plt.show()
plt.plot(history.history["val_loss"],color="g",label="val_loss")
plt.plot(history.history["loss"],color="y",label="train_loss")
plt.title("loss graph")
plt.xlabel("number ofepoch")
plt.ylabel("loss")
plt.legend()
plt.grid
plt.show()




model.evaluate(x_val,y_val)



category_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'K',10:'L',11:'M',12:'N',13:'O',14:'P',15:'Q',16:'R',17:'S',18:'T',19:'U',20:'V',21:'W',22:'X',23:'Y'}
pred_list=[]
plt.figure()
plt.subplots(figsize=(6,5))
for i in range(0,9):
  plt.subplot(3,3,i+1)
  plt.imshow(X_test[i])
  #prediction
  predict_x=model.predict(X_test[0:10])
  classes_x=np.argmax(predict_x,axis=1)
  pred_list.append(classes_x)
  #display predicted labels of images
  plt.title("output: "+category_dict[classes_x[i]])
  plt.axis("off")
plt.show()




path=r"/content/drive/MyDrive/"
image=os.path.join(path,'60.jpg')
if os.path.exists(image):
  img=cv2.imread(image)
  img_size=100
  img=cv2.resize(img,(img_size,img_size))
  img=img.reshape(-1,img_size,img_size,3)
  img=img/255
  predict_x=model.predict(img)
  classes_x=np.argmax(predict_x,axis=1)
  
   # Print the predicted class probabilities
  for i, prob in enumerate(predict_x[0]):
      print(f"Class {i}: {prob}")

  classes_x = np.argmax(predict_x, axis=1)
  print(category_dict[classes_x[0]])
else : print("image doesn't exists")



import matplotlib.pyplot as plt
letters=["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
# Initialize a list to store the recognition percentages
recognition_percentages = []

# Print the predicted class probabilities and store the recognition percentages
for i, prob in enumerate(predict_x[0]):
    class_label = category_dict[i]
    recognition_percentage = round(prob*100, 2)
    recognition_percentages.append(recognition_percentage)
    print(f"Class {class_label}: {recognition_percentage}%")

# Plot the recognition percentages as a bar chart
fig, ax = plt.subplots()
ax.set_title('Percentage of Letter Recognition')
ax.set_xlabel('Letters')
ax.set_ylabel('Percentage')
ax.bar(letters, recognition_percentages)
plt.show()





y_pred=model.predict(x_val)
y_pred_classes=np.argmax(y_pred, axis=1)
y_true=np.argmax(y_val, axis=1)
conf_mat=confusion_matrix(y_true,y_pred_classes)
letters_names=list('abcdefghiklmnopqrstuvwxy')
f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(conf_mat, annot=True,fmt='.0f')
ax.set_xticklabels(letters_names)
ax.set_yticklabels(letters_names)
plt.show()
#plot the confusion matrix for the Testing data



y_pred2=model.predict(X_test)
y_pred_classes2=np.argmax(y_pred2,axis=1)
y_true2=np.argmax(Y_test,axis=1)
conf_mat2=confusion_matrix(y_true2,y_pred_classes2)
f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(conf_mat2,annot=True,fmt=".0f")
ax.set_xticklabels(letters_names)
ax.set_yticklabels(letters_names)
plt.show()