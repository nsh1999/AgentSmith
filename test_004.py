import os

from typesafe_sdk import Noul, TypeSafeClient

client = TypeSafeClient(
    api_key='lmstudio',
    base_url="http://localhost:1234",
)

response = client.system_one(
    state="Hello!",
    model="semif-qwen3.5-4b",
#    model="semif-qwen3.5-4b-mlx",
    questions={
        "is_helpful": Noul(
            instructions="Does this explain what an LLM gateway does?"
        ),
    },
)

