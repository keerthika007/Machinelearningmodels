import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
dataset = pd.read_csv("C:/Users/keert/MyNewFolder/LinearRegression/spam.csv", encoding="latin-1")

# Drop unnecessary columns
dataset = dataset[['v1', 'v2']].dropna()

# Encode labels: 'ham' -> 0, 'spam' -> 1
dataset['label'] = dataset['v1'].map({'ham': 0, 'spam': 1})

# Extract features (text data) and labels
a = dataset['v2']  # Assuming 'v2' contains the message text
b = dataset['label']

# Convert text to numerical features using TF-IDF Vectorization
vectorizer = TfidfVectorizer()
a_transformed = vectorizer.fit_transform(a)

# Split data into training and testing sets
a_train, a_test, b_train, b_test = train_test_split(a_transformed, b, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(a_train, b_train)

# Predictions
b_pred = model.predict(a_test)

# Evaluate model accuracy
accuracy = accuracy_score(b_test, b_pred)
print(f"Model Accuracy: {accuracy:.2f}")
