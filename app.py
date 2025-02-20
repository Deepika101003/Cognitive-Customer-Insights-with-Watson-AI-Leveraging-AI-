from flask import Flask, request, render_template
import os
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EntitiesOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize IBM Watson NLU
authenticator = IAMAuthenticator(os.getenv('NLU_API_KEY'))
nlu = NaturalLanguageUnderstandingV1(
    version='2023-03-15',
    authenticator=authenticator
)
nlu.set_service_url(os.getenv('NLU_SERVICE_URL'))

def analyze_text(text):
    """Analyze text using IBM Watson NLU."""
    response = nlu.analyze(
        text=text,
        features=Features(
            sentiment=SentimentOptions(),
            entities=EntitiesOptions()
        )
    ).get_result()
    return response

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze customer feedback and return formatted output."""
    text = request.form['text']
    analysis = analyze_text(text)

    # Determine sentiment label based on score
    sentiment_score = analysis['sentiment']['document']['score']
    if sentiment_score >= 0.5:
        sentiment_label = "positive"
    elif sentiment_score <= -0.5:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # Format the output
    formatted_output = {
        "analysis": {
            "sentiment": sentiment_label,
            "confidence": abs(sentiment_score),  # Use absolute value for confidence
            "entities": [
                {
                    "name": entity['text'],
                    "type": entity['type'],
                    "relevance": entity['relevance']
                }
                for entity in analysis['entities']
            ]
        }
    }

    return formatted_output

if __name__ == '__main__':
    app.run(debug=True)