import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from keras.models import Sequential,load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\customer_churn_dataset-training-master.csv")
test_data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\customer_churn_dataset-testing-master.csv")

print("Training Data: ")
print(train_data.head())
print("Testing Data: ")
print(test_data.head())

train_data = train_data.dropna().reset_index(drop=True)

print("Missing values: ",train_data.isnull().sum())
print("Missing values: ",test_data.isnull().sum())

oe = OneHotEncoder(drop='first',sparse_output=False)
train_encoded_df = pd.DataFrame(oe.fit_transform(train_data[["Gender","Subscription Type","Contract Length"]]),columns=oe.get_feature_names_out(["Gender","Subscription Type","Contract Length"]),index=train_data.index)
train_data = train_data.drop(["CustomerID","Gender","Subscription Type","Contract Length"],axis=1)
train_data = pd.concat([train_data,train_encoded_df],axis=1)

test_encoded_df = pd.DataFrame(oe.transform(test_data[["Gender","Subscription Type","Contract Length"]]),columns=oe.get_feature_names_out(["Gender","Subscription Type","Contract Length"]),index=test_data.index)
test_data = test_data.drop(["CustomerID","Gender","Subscription Type","Contract Length"],axis=1)
test_data = pd.concat([test_data,test_encoded_df],axis=1)

X_train = train_data.drop("Churn",axis=1)
y_train = train_data["Churn"]

X_test = test_data.drop("Churn",axis=1)
y_test = test_data["Churn"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_final,X_val,y_train_final,y_val = train_test_split(
    X_train,y_train,test_size=0.2,random_state=42
)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_final,y_train_final))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val,y_val))

train_dataset = train_dataset.shuffle(len(X_train_final)).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

model = Sequential([
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])

earlystop = EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
model_check = ModelCheckpoint(filepath="best_model.keras",monitor="val_loss",save_best_only=True)

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(train_dataset,validation_data=val_dataset,epochs=30,callbacks=[earlystop,model_check])

pred = model.predict(X_test)
pred = (pred>=0.5).astype(int)
print(pred[:5])

loss,acc = model.evaluate(X_test,y_test)
print("Loss: ",loss)
print("Accuracy: ",acc)

accuracy = accuracy_score(y_test,pred)
print("Accuracy: ",accuracy)

precision = precision_score(y_test,pred)
print("Precision: ",precision)

recall = recall_score(y_test,pred)
print("Recall: ",recall)

f1 = f1_score(y_test,pred)
print("F1 Score: ",f1)

cm = confusion_matrix(y_test,pred)
plt.title("Confusion Matrix: ")
sns.heatmap(cm,annot=True,xticklabels=["No Churn","Churn"],yticklabels=["No Churn","Churn"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

report = classification_report(y_test,pred)
print("Classification report: ")
print(report)

best_model = load_model("best_model.keras")
loss_final,acc_final = best_model.evaluate(X_test,y_test)
print("Loss: ",loss_final)
print("Accuracy: ",acc_final)