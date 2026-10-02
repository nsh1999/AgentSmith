from typesafe_sdk import TypeSafeClient, Choice, Noul

# 1. Point the official TypeSafe SDK to your local SemIf server
client = TypeSafeClient(
    api_key="local",
    base_url="http://127.0.0.1:1234",  # Local SemIf instance
    model="semif-qwen3.5-4b-mlx"
)

# 2. Pass in the raw application state
state = "The user is complaining that their credit card was charged twice on checkout."

# 3. Define the strict, typed choices you need back
questions = {
    "is_billing": Noul(instructions="Is this ticket about a billing issue?"),
    "priority": Choice(
        instructions="What is the severity level?",
        criteria={"low": "Minor issue", "high": "Urgent/Loss of funds"}
    )
}

# 4. Execute the single-forward pass decision
response = client.system_one(state=state, questions=questions)

print(response["is_billing"].probability)
print(response["priority"].selected)

