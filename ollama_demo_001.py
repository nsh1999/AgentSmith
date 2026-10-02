import os
import sys
import time
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

os.environ["TYPESAFE_API_KEY"] = "ollamna"
os.environ["TYPESAFE_BASE_URL"] = "http://localhost:11434"
os.environ["TYPESAFE_DEFAULT_MODEL"] = "nimble:latest"

# "hf.co/openjev/openjev-GGUF:latest",
ticket = "I was charged twice. Please refund the extra payment."
models = [ "tev1:4b", "nimble:latest"]
questions = {
    "team": Choice(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "Payments and refunds",
            "technical": "Bugs and integrations",
            "other": "None of the above",
        },
    ),
    "refund": Noul(
        instructions="Does the customer explicitly ask for a refund?",
    ),
    "urgency": Score(
        instructions="How urgent is this ticket?",
        criteria=["Routine", "Soon", "Urgent"],
    ),
}

for model in models:
    print('----------------------------------------')
    print(f"Available model: {model}")
    start_time = time.time()
    with TypeSafeClient(timeout=120) as client:
        result = client.system_one(
            model=model,
            state={"ticket": ticket},
            questions=questions,
        )
    end_time = time.time()
    print(f"Time taken for model {model}: {int(end_time - start_time) // 60}:{int(end_time - start_time) % 60}.{int((end_time - start_time) * 1000) % 1000} minutes:seconds.milliseconds")
    print("TEAM:", result.choices["team"].choice)   # billing
    print("REFUND:", result.nouls["refund"].noul)     # 0.997
    print("URGENCY:", result.scores["urgency"].score)  # 0.69
    #print("RAW RESULT:", result)  # Print the entire result object for debugging
    print()

