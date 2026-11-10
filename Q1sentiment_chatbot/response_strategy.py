class ResponseStrategy:

    def generate_response(self, user_message, sentiment):

        if sentiment == "Positive":

            return (
                "That's wonderful to hear! 😊 "
                "I'm glad you're having a positive experience. "
                "How else can I assist you today?"
            )

        elif sentiment == "Negative":

            return (
                "I'm sorry to hear that you're experiencing this. 😔 "
                "I understand your concern, and I'll do my best to help. "
                "Could you tell me more about the problem?"
            )

        else:

            return (
                "I understand. 🙂 "
                "Thank you for sharing that with me. "
                "How can I assist you further?"
            )


if __name__ == "__main__":

    strategy = ResponseStrategy()

    print("\n===== RESPONSE STRATEGY TEST =====")

    test_messages = [
        ("I am very happy with the service!", "Positive"),
        ("I am disappointed with the service.", "Negative"),
        ("The service is okay.", "Neutral")
    ]

    for message, sentiment in test_messages:

        response = strategy.generate_response(
            message,
            sentiment
        )

        print("\nUser:", message)
        print("Sentiment:", sentiment)
        print("Bot:", response)