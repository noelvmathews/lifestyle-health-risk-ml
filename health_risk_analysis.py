import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

print("--- Lifestyle-Based Health Risk Analysis ---")

# 1. Theoretical Centroid Baseline from the Assignment
# In Section 2, k=3 clustering isolates clear risk profiles
low_risk_centroid = {"BMI": 22, "ExerciseHours": 5}
print(f"Target Low-Risk Cluster Centroid: {low_risk_centroid}")

# Centroid calculation logic demo: Mean of [22, 24, 26, 28] -> 25
sample_cluster = np.array([22, 24, 26, 28])
updated_centroid_bmi = np.mean(sample_cluster)
print(f"Centroid Update Step Verification: Mean BMI = {updated_centroid_bmi}")

print("\n--- Decision Tree Root Node Inference ---")
# Section 3 Split Analysis: BMI < 27
total_samples = 200
yes_branch_samples = 117
no_branch_samples = 83

# Class counts inside the Yes Branch (BMI < 27)
low_risk_count = 70
medium_risk_count = 47
high_risk_count = 0

# Probabilities
p_low = low_risk_count / yes_branch_samples
p_med = medium_risk_count / yes_branch_samples
p_high = high_risk_count / yes_branch_samples

# Gini Impurity Calculation: Gini = 1 - (p_low^2 + p_med^2 + p_high^2)
gini_impurity = 1 - (p_low**2 + p_med**2 + p_high**2)
print(f"Total samples passing root split (BMI < 27): {yes_branch_samples}")
print(f"Calculated Node Gini Impurity: {gini_impurity:.2f}")
print("Conclusion: Node is highly impure; further feature splitting is required.")
