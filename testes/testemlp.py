# baseado em https://www.kaggle.com/ahmethamzaemra/mlpclassifier-example
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#1. Faz a leitura do arquivo e guarda em um Dataframe
df = pd.read_csv('Dataset_spine.csv')

# Descreve detalhes do Dataset
df.head()
df.describe()

#2. Define as colunas
colunas = df.columns
colunas_features = colunas[0:12]
colunas_label = colunas[12]

#3. Dados em Dataframe para X e Y
X_df = df[colunas_features]
y_df = df[colunas_label]

#4. Transforma Datafram em array
X = X_df.values
y = y_df.values

#5. Etapa de definição de conjuntos de treinamento e conjuntos de testes
x_train, x_test, y_train, y_test = train_test_split(X_df,y_df, test_size= 0.25, random_state=27)

#6. Cria um modelo MultiLayerPerceptron
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

#7. Ajusta os dados do modelo aos dados de treinamento
clf.fit(x_train, y_train)

#8. Faz a prediction baseada no conjunto de testes
y_pred = clf.predict(x_test)

print("x_test: ", x_test)
print("Predict de x_test", y_pred)

#Acuracia do modelo proposto
accuracy_score(y_test, y_pred)

#Matriz de confusao
cm = confusion_matrix(y_test, y_pred)

print("Matriz de confusao", cm)
