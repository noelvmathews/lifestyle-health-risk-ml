import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("--- Lifestyle-Based Health Risk Analysis Pipeline ---")

# 1. Simulate a realistic sample dataset matching your overview metrics
np.random.seed(42)
n_samples = 200

bmi = np.random.uniform(18.5, 35, n_samples)
exercise = np.random.uniform(0, 10, n_samples)

# Generate targets based on your rules (BMI < 27 and Exercise patterns)
risk_labels = []
for b, e in zip(bmi, exercise):
    if b < 27:
        risk_labels.append("Low" if e > 4 else "Medium")
    else:
        risk_labels.append("High" if e < 2 else "Medium")

df = pd.DataFrame({"BMI": bmi, "ExerciseHours": exercise, "HealthRisk": risk_labels})

# 2. Section 2: K-Means Clustering (With scaling)
X_cluster = df[["BMI", "ExerciseHours"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
print("✅ Unsupervised K-Means clustering executed successfully.")

# 3. Section 3: Supervised Machine Learning (Decision Tree with Regularization)
X = df[["BMI", "ExerciseHours"]]
y = df["HealthRisk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.3, random_state=42)

# Enforcing max_depth=3 to stop the 25% overfitting issue you noted!
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f"✅ Decision Tree Trained. Training Accuracy: {train_acc:.2%}")
print(f"✅ Generalized Test Accuracy (Post-Regularization): {test_acc:.2%}")

# Print out the literal rules of your tree to prove the split conditions
tree_rules = export_text(clf, feature_names=["BMI", "ExerciseHours"])
print("\n--- Model Decision Paths ---")
print(tree_rules)
