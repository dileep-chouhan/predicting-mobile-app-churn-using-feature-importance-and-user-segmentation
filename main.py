import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_users = 500
data = {
    'Daily_Sessions': np.random.randint(1, 5, size=num_users),
    'Avg_Session_Duration': np.random.uniform(5, 60, size=num_users),
    'App_Version': np.random.choice(['v1', 'v2', 'v3'], size=num_users),
    'Notifications_Enabled': np.random.choice([True, False], size=num_users),
    'Churn': np.random.choice([0, 1], size=num_users, p=[0.8, 0.2]) # 20% churn rate
}
df = pd.DataFrame(data)
# --- 2. Data Preprocessing ---
# One-hot encode categorical features
df = pd.get_dummies(df, columns=['App_Version'], drop_first=True)
# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']
# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 3. Model Training and Evaluation ---
# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# --- 4. Feature Importance ---
feature_importances = model.feature_importances_
feature_names = df.drop('Churn', axis=1).columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Plot saved to feature_importance.png")
# --- 6. User Segmentation (Example using k-means -  requires further refinement for real-world application)---
# This section is a simplified example and would need more sophisticated clustering in a real project.
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42) # Choosing 3 clusters as an example.  Needs further analysis to determine optimal k.
kmeans.fit(X)
df['Segment'] = kmeans.labels_
#Analyze churn rate within segments (example)
churn_by_segment = df.groupby('Segment')['Churn'].mean()
print("\nChurn rate by segment:")
print(churn_by_segment)
#Further analysis of segments would be needed to tailor retention strategies.