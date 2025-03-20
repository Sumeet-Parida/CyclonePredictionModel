# CyclonePredictionModel
Cyclone prediction model using machine learning algorithms .
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


file_path = "odisha_cyclone_data_1000_days.csv"
df = pd.read_csv(file_path)


label_encoder = LabelEncoder()
df['District'] = label_encoder.fit_transform(df['District'])


X = df.drop(['Date', 'District'], axis=1)  
y = df['District']  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVC": SVC(kernel='linear')
}

results = {}
print("\n Model Performance:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


future_data = np.array([
    [180, 940, 30.0, 250, 80, 4.0, 15000, 50, 1200000, 12, 3000, 6000]  
])

future_data_scaled = scaler.transform(future_data)


best_model_name = max(results, key=results.get)
best_model = models[best_model_name]


predicted_district = best_model.predict(future_data_scaled)
district_name = label_encoder.inverse_transform(predicted_district)[0]

print(f"\nFuture cyclone is predicted to hit: {district_name}")
print(f"\n Best-performing model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")

plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


district_coords = {
    "Angul": (20.8, 85.1), "Balangir": (20.7, 83.5), "Balasore": (21.5, 86.9),
    "Bargarh": (21.3, 83.6), "Bhadrak": (21.1, 86.5), "Boudh": (20.8, 84.3),
    "Cuttack": (20.5, 85.9), "Deogarh": (21.5, 84.7), "Dhenkanal": (20.7, 85.6),
    "Gajapati": (18.9, 84.1), "Ganjam": (19.4, 84.8), "Jagatsinghpur": (20.3, 86.3),
    "Jajpur": (20.8, 86.3), "Jharsuguda": (21.8, 84.0), "Kalahandi": (19.9, 83.2),
    "Kandhamal": (20.1, 84.0), "Kendrapara": (20.5, 86.4), "Kendujhar": (21.6, 85.6),
    "Khurdha": (20.1, 85.6), "Koraput": (18.8, 82.7), "Malkangiri": (18.3, 81.9),
    "Mayurbhanj": (21.9, 86.7), "Nabarangpur": (19.3, 82.5), "Nayagarh": (20.1, 85.1),
    "Nuapada": (20.0, 82.5), "Puri": (19.8, 85.8), "Rayagada": (19.2, 83.4),
    "Sambalpur": (21.4, 84.0), "Subarnapur": (20.8, 83.9)
}


plt.figure(figsize=(10, 10))
plt.title(f"Predicted Cyclone Hit District: {district_name}", fontsize=14)


for district, coords in district_coords.items():
    x, y = coords
    plt.scatter(x, y, color='blue', label='District' if district == "Angul" else "")
    plt.text(x + 0.1, y + 0.1, district, fontsize=9)


pred_x, pred_y = district_coords[district_name]
plt.scatter(pred_x, pred_y, color='red', label='Predicted District', s=200, edgecolors='black')

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.grid(True)
plt.show()


