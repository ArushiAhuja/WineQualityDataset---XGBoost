import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


from google.colab import files
uploaded = files.upload()

df = pd.read_excel('wine.xlsx')
#displaying the first 5 rows of the dataset 
df.head()

label = LabelEncoder()
df["wine_quality"] = label.fit_transform(df["wine_quality"])

X = df.drop("wine_quality", axis=1)
y = df["wine_quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Bad","Good"]))

new_wine = np.array([[7.2, 0.65, 0.03, 2.0, 0.08, 18, 45, 0.9972, 3.30, 0.60, 9.6]])
prediction = model.predict(new_wine)

print("Wine Quality Prediction:", label.inverse_transform(prediction)[0])

cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True,fmt='d')
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

