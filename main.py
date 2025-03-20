from transformers import BloomTokenizerFast, BloomModel
import torch

tokenizer = BloomTokenizerFast.from_pretrained("./tokenizers/bloom")
model = BloomModel.from_pretrained("./models/bloom")

prompt = "LLMs are great tools to predict the behaviour of people during a pandemic."
result_length = 50
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"],
                         max_length=result_length,
                         do_sample=True,
                         top_k=50,
                         top_p=0.9)
answer = tokenizer.decode(outputs[0])

print(answer)
