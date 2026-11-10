from sentiment import SentimentAnalyzer
from response_strategy import ResponseStrategy


class SentimentChatbot:

    def __init__(self):
        print("Initializing chatbot...")

        self.sentiment_analyzer = SentimentAnalyzer()
        self.response_strategy = ResponseStrategy()

        print("Chatbot initialized successfully!")

    def get_response(self, user_message):

        # Analyze the user's sentiment
        sentiment_result = self.sentiment_analyzer.analyze(
            user_message
        )

        sentiment = sentiment_result["sentiment"]
        confidence = sentiment_result["confidence"]

        # Generate response based on sentiment
        response = self.response_strategy.generate_response(
            user_message,
            sentiment
        )

        return {
            "message": user_message,
            "sentiment": sentiment,
            "confidence": confidence,
            "response": response
        }


if __name__ == "__main__":

    print("\n======================================")
    print("       SENTIMENT-AWARE CHATBOT")
    print("======================================")

    chatbot = SentimentChatbot()

    while True:

        user_message = input(
            "\nYou (type 'exit' to quit): "
        )

        if user_message.lower().strip() == "exit":
            print("\nBot: Goodbye! Have a great day! 👋")
            break

        result = chatbot.get_response(user_message)

        print("\n--------------------------------------")
        print("Sentiment  :", result["sentiment"])
        print("Confidence :", result["confidence"])
        print("Bot        :", result["response"])
        print("--------------------------------------")