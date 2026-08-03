from keras import utils
import tensorflow as tf
from keras.layers import RandomFlip,RandomRotation,RandomZoom,Dense,MaxPooling2D,Conv2D,Flatten,Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint

train_data = utils.image_dataset_from_directory(r"C:\Users\ASUS\Downloads\kagglecatsanddogs_5340\PetImages",label_mode="binary",color_mode="rgb",validation_split=0.2,seed=42,subset="training",batch_size=32,image_size=(128,128))
validation_data = utils.image_dataset_from_directory(r"C:\Users\ASUS\Downloads\kagglecatsanddogs_5340\PetImages",label_mode="binary",color_mode="rgb",validation_split=0.2,seed=42,subset="validation",batch_size=32,image_size=(128,128))

train_data = train_data.map(lambda x,y : (tf.cast(x,tf.float32)/255.0,y))
validation_data = validation_data.map(lambda x,y : (tf.cast(x,tf.float32)/255.0,y))

train_data = train_data.ignore_errors()
validation_data = validation_data.ignore_errors()

train_data = train_data.prefetch(tf.data.AUTOTUNE)
validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

model = Sequential([
    Input(shape=(128,128,3)),
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    Conv2D(32,kernel_size=(3,3),padding="same",activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64,kernel_size=(3,3),padding="same",activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128,kernel_size=(3,3),padding="same",activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128,activation="relu"),
    Dense(1,activation="sigmoid")
])

print(model.summary())

earlystop = EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
ModelCheck = ModelCheckpoint(filepath="best_model.keras",monitor="val_loss",save_best_only=True)

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(train_data,validation_data=validation_data,epochs=30,callbacks=[earlystop,ModelCheck])

for images, labels in validation_data.take(1):
    pred = model.predict(images[:5])
    pred = (pred >= 0.5).astype(int)

    print("Prediction:")
    print(pred)

    print("Actual:")
    print(labels[:5].numpy())

    break

loss,acc = model.evaluate(validation_data)
print("Loss: ",loss)
print("Accuracy: ",acc)