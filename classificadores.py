import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

url = "C:/Users/Paula/Documents/avaliacao_classificadores/wdbc.csv"
column_names = ["ID", "radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1", "concavity1",
                "concave_points1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2",
                "smoothness2", "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
                "radius3", "texture3", "perimeter3", "area3", "smoothness3", "compactness3", "concavity3",
                "concave_points3", "symmetry3", "fractal_dimension3", "Diagnosis"]

data = pd.read_csv(url, header=0, names=column_names)

# Separar características (X) e rótulos (y)
X = data.drop(["ID", "Diagnosis"], axis=1)
y = data["Diagnosis"]

# Normalizar apenas as características
scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Treinar o modelo Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Salvar o modelo
joblib.dump(tree, 'decision_tree_model.joblib')

# Gerar a matriz de confusão
y_pred = tree.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Calcular a taxa de erro e a taxa de acertos
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

print(f"Acurácia: {accuracy * 100:.2f}%")
print(f"Taxa de Erro: {error_rate * 100:.2f}%")