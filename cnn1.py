from tensorflow import keras

(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()
X_train=X_train.reshape(-1,28,28,1)/255
y_train=keras.utils.to_categorical(y_train)
X_test=X_test.reshape(-1,28,28,1)/255
y_test=keras.utils.to_categorical(y_test)
print(y_train[:5])
print('loading complete')

model=keras.Sequential()
model.add(keras.layers.Input(shape=(28,28,1,)))
model.add(keras.layers.Conv2D(24,kernel_size=5,activation='relu',padding='same'))
model.add(keras.layers.MaxPool2D(padding='same'))
model.add(keras.layers.Conv2D(48,kernel_size=5,activation='relu',padding='same'))
model.add(keras.layers.MaxPool2D(padding='same'))
model.add(keras.layers.Conv2D(64,kernel_size=5,activation='relu',padding='same'))
model.add(keras.layers.MaxPool2D(padding='same'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
print("compiling done")

model.fit(X_train,y_train,epochs=10,use_multiprocessing=True)
print('fit done')
evaluate=model.evaluate(X_train,y_train,return_dict=True)
print(evaluate)