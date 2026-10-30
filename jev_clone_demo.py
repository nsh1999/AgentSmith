from decision_maker import decision_maker,resutlt_set_to_html,majority_2_3,explain 
from typesafe_sdk import Choice, Noul, Score
from pprint import pprint
from IPython.display import HTML, display

models = [
    #"pplx-decider:27b-nvfp4",
    "clef-flash:9b-mxfp8",
    #"clef:27b-nvfp4",
    "kev:9b-nvfp4",
    #'laya:421m-typed-decisions-mlx-fp16',
    "nimble-v3:9b-nvfp4",
    "strands-decider:2b-mlx-bf16",
    "tev1:4b-mxfp8",
]

ticket = "I was charged twice. Please refund the extra payment. this is urgent for me."
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
result_set = decision_maker(models, questions, ticket)

explain(result_set, questions)
for key in questions:
    print(f"{key}\t{questions[key].type}\t{result_set[key]['majority']}")
result_set["models"]

