# watson.py
import pandas as pd
import numpy as np
from preprocessing import preprocess_data  # Import preprocessing functions
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv(r"C:\Users\ASUS\Downloads\ecommerce_data .csv")

# Preprocess the data
data = preprocess_data(data)

# Save cleaned data
data.to_csv("cleaned_customer_data.csv", index=False)
print("Data cleaned and saved!")

# Data Analysis
# Customer age distribution
sns.histplot(data["customer_age"], bins=10, kde=True)
plt.title("Customer Age Distribution")
plt.show()

# Purchase amount by product category
sns.barplot(data=data, x="product_category", y="purchase_amount", errorbar=None)
plt.title("Purchase Amount by Product Category")
plt.xticks(rotation=45)
plt.show()

# Loyalty status breakdown
loyalty_counts = data["loyalty_status"].value_counts()
loyalty_counts.plot(kind="pie", autopct="%1.1f%%")
plt.title("Loyalty Status Distribution")
plt.show()

# Machine Learning
# Create target and features
data["churn"] = np.where(data["loyalty_status"] == "Standard", 1, 0)  # Assume "Standard" loyalty indicates churn
X = data[["customer_age", "purchase_amount", "product_rating"]]
y = data["churn"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Sentiment Analysis with IBM Watson NLU
# Replace 'YOUR_API_KEY' and 'YOUR_URL' with your actual API key and URL.
apikey = os.getenv('NLU_API_KEY')
url = os.getenv('NLU_SERVICE_URL')

authenticator = IAMAuthenticator(apikey)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(url)

# Analyze customer feedback
customer_feedback = [
    "I love the new features of the product!",
    "The service was terrible; I had to wait too long.",
    "Not what I expected, but the support team helped me.",
    "Fantastic experience overall! I will recommend it to others."
]

def analyze_feedback(feedback):
    overall_sentiment = {'positive': 0, 'neutral': 0, 'negative': 0}
    keywords_summary = {}

    for text in feedback:
        try:
            response = nlu.analyze(
                text=text,
                features={
                    'sentiment': {},
                    'keywords': {}
                }
            ).get_result()

            # Extract sentiment
            sentiment = response['sentiment']['document']['label']
            overall_sentiment[sentiment] += 1

            # Extract keywords
            keywords = response['keywords']
            for keyword in keywords:
                word = keyword['text']
                if word in keywords_summary:
                    keywords_summary[word] += 1
                else:
                    keywords_summary[word] = 1

        except ApiException as e:
            print(f"Error: {e}")

    print("Overall Sentiment Summary:")
    print(overall_sentiment)
    print("\nKeywords Summary:")
    print(keywords_summary)

# Call the function to analyze feedback
analyze_feedback(customer_feedback)