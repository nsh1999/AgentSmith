from typesafe_sdk import TypeSafeClient

# Points to a custom Jev/SemIf-compatible endpoint
client = TypeSafeClient(
    api_key="your-key",
    base_url="http://127.0.0.1:1234"
)

response = client.system_one(
    state="My running shoes arrived in the wrong size. Can I swap them?",
    questions={
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "returns": "Exchanges, refunds, or damaged items",
                "shipping": "Delivery status and tracking",
                "billing": "Charges and invoices"
            }
        }
    }
)

print(response.choices["department"].choice) # Outputs: "returns"

