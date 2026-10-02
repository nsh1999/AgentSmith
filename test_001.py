import lmstudio as lms

model = lms.llm("qwen3.8-27b-mlx")
result = model.respond("there are 3 killers in the room. A new person enters the room and kills one of them. Nobody leaves the room. How many killers are there in the room?")

print(result)

