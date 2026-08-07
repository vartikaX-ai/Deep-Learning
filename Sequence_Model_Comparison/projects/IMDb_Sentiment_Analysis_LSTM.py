import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import LSTM,Embedding,Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\IMDB_Dataset_CLEANED.csv")

X = df["review"]
y = df["sentiment"]

le = LabelEncoder()
y = pd.Series(le.fit_transform(y),name="sentiment")

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train,maxlen=200)

vocab_size = len(tokenizer.word_index)+1

model = Sequential([
    Embedding(vocab_size,128),
    LSTM(64,return_sequences=True),
    LSTM(32),
    Dense(16,activation="relu"),
    Dense(1,activation='sigmoid')
])

earlystop = EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
modelcheck = ModelCheckpoint(filepath="best_model.keras",monitor="val_loss",save_best_only=True)

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(X_train,y_train,epochs=5,callbacks=[earlystop,modelcheck],validation_split=0.2)

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test,maxlen=200)

pred = model.predict(X_test)
pred = (pred>0.5).astype(int)
print(pred)

loss,acc = model.evaluate(X_test,y_test)
print("Loss: ",loss)
print("Accuracy: ",acc)