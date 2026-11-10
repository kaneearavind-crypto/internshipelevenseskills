from transformers import pipeline


class SentimentAnalyzer:

    def __init__(self):
        print("Loading 3-class sentiment analysis model...")

        self.classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

        print("Model loaded successfully!")

    def analyze(self, text):

        text = text.strip()

        if not text:
            return {
                "sentiment": "Neutral",
                "confidence": 0.0
            }

        result = self.classifier(text)[0]

        label = result["label"].lower()
        confidence = result["score"]

        # Neutral expressions commonly used in customer conversations
        neutral_phrases = [
            "okay",
            "ok",
            "fine",
            "average",
            "normal",
            "acceptable",
            "nothing special",
            "neither good nor bad",
            "it's okay",
            "it is okay",
            "it's fine",
            "it is fine"
        ]

        text_lower = text.lower()

        # Handle clearly neutral expressions
        if any(phrase in text_lower for phrase in neutral_phrases):
            sentiment = "Neutral"
        elif label == "positive":
            sentiment = "Positive"
        elif label == "negative":
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4)
        }


if __name__ == "__main__":

    print("\n====================================")
    print("   SENTIMENT ANALYSIS TEST")
    print("====================================\n")

    analyzer = SentimentAnalyzer()

    while True:

        text = input(
            "\nEnter a message (or type 'exit' to quit): "
        )

        if text.lower().strip() == "exit":
            print("\nExiting sentiment analyzer...")
            break

        result = analyzer.analyze(text)

        print("\n------------------------------------")
        print("Message    :", text)
        print("Sentiment  :", result["sentiment"])
        print("Confidence :", result["confidence"])
        print("------------------------------------")