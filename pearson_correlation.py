import numpy as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("StudentPerformance-encoded.csv", encoding='latin1')
df.head(15)

df.info()

"""**Filter Method menggunakan Pearson Correlation**"""

cor = df.corr()
plt.figure(figsize = (24,22))
sns.heatmap(cor, cmap = 'Blues', annot = True)
plt.show()

cor_target = abs(cor["lulus_tepatwaktu"])

relevant_features = cor_target[cor_target>=0.5]
relevant_features

"""**Data Split**"""

X = df.iloc[:, [1, 15, 18,19, 22, 29]].values
y= df.iloc[:, -1].values

#menampilkan nilai x dan y
print('X \n',X[:5])
print('\ny \n',y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state= 21)

print(X_train.shape)
print (y_train.shape)
print(X_test.shape)
print (y_test.shape)

"""**Create classifier model**"""

# Training the Naive Bayes model on the Training set
classifier_model_binary = GaussianNB()
classifier_model_binary.fit (X_train, y_train)

"""**Predict test data**"""

y_pred = classifier_model_binary.predict(X_test)

y_pred

y_test

classifier_model_binary.predict_proba(X_test)

def predict_one(persentase_presensi, ipk):
  print('Prediction: ',classifier_model_binary.predict([[persentase_presensi, ipk]]))
  print('Probability: ',classifier_model_binary. predict_proba([[persentase_presensi, ipk]]))

def predict_one(age, persentase_presensi, durasi_sosmed, jumlah_sosmed, score_toefl, ipk):
  features = [[age, persentase_presensi, durasi_sosmed, jumlah_sosmed, score_toefl, ipk]]

"""**Evaluate**"""

print(classification_report (y_test,y_pred, zero_division=0))

cm = confusion_matrix (y_test, y_pred)
cm

from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay (confusion_matrix=cm,
                                display_labels=classifier_model_binary.classes_)
disp.plot()
plt.show()