import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlp import MLP
from metricas import acuracia, mse, validacao_cruzada, matriz_confusao
from sklearn.preprocessing import OneHotEncoder


# Carregar o dataset de caracteres
base_dir = os.path.dirname(__file__)
X = np.load(os.path.join(base_dir, 'X.npy'))
y = np.load(os.path.join(base_dir, 'Y_classe.npy'))

print("Shape original de X:", X.shape)

X = X.reshape(X.shape[0], -1)  # Flatten
print("Shape ajustado de X para a MLP:", X.shape)

# Normalizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding dos rótulos
onehot_encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = pd.get_dummies(y_train).values

print("Shape de X_train:", X_train.shape)
print("Shape de y_train_onehot:", y_train_onehot.shape)

# Parâmetros da MLP
input_size = X_train.shape[1]
hidden_layers = 5
output_size = len(np.unique(y))

mlp = MLP(input_size, hidden_layers, output_size, taxa_aprendizado=0.01, epocas=20000)

print("Iniciando o treinamento...")
errors = mlp.fit(X_train, y_train_onehot)

# Predição
y_pred_probs = mlp.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Avaliação
acc = acuracia(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {acc * 100:.2f}%")

# Validação cruzada
validacao_cruzada(
    x_treino=X_train,
    k_folds=5,
    y_treino=y_train_onehot,
    model_params={
        "tamanho_entrada": input_size,
        "camadas_escondidas": hidden_layers,
        "tamanho_saida": output_size,
        "taxa_aprendizado": 0.01,
        "epocas": 20000
    }
)

matriz_confusao(y_test, y_pred)
mlp.relatorio_final(errors, X_test, y_test)
print("Treinamento e teste concluídos.")