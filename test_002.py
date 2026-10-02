from openai import OpenAI

try:
	client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
	completion = client.chat.completions.create(
	#model="ibm/granite-4-h-small",
	model="qwen/qwen3.8-27b",
	messages=[
	{"role": "system", "content": "Always answer in rhymes."},
	{"role": "user", "content": "there are 3 killers in the room. A new person enters the room and kills one of them. nobody leaves the room. how many killers are there in the room?"}
	],
	temperature=0.7,
	)
	print(completion.choices[0].message.reasoning_content)
except Exception as ex:
	print('oooops')
	print(ex)


