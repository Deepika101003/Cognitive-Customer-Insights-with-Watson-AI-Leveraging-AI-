# preprocessing.py
import pandas as pd
import numpy as np
import re

def clean_text(text):
    """Clean text by removing special characters, HTML tags, and extra spaces."""
    if isinstance(text, str):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra spaces
        text = ' '.join(text.split())
    return text

def preprocess_data(data):
    """Preprocess the dataset."""
    # Rename columns
    data.columns = [
        "customer_id", "customer_name", "product_category", "purchase_amount",
        "delivery_status", "payment_status", "customer_age", "customer_gender",
        "product_rating", "shipping_region", "loyalty_status", "country"
    ]

    # Handle missing values
    data.fillna({
        "product_category": "Unknown",
        "purchase_amount": data["purchase_amount"].mean(),
        "customer_gender": "Unknown",
        "loyalty_status": "Standard",
        "country": "Unknown"
    }, inplace=True)

    # Convert columns to appropriate types
    data["purchase_amount"] = data["purchase_amount"].astype(float)
    data["customer_age"] = data["customer_age"].fillna(data["customer_age"].mean()).astype(int)

    # Clean text columns
    text_columns = ["customer_name", "product_category", "delivery_status", "payment_status", "shipping_region", "country"]
    for col in text_columns:
        data[col] = data[col].apply(clean_text)

    return data