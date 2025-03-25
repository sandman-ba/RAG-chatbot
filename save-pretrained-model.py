from transformers import BloomTokenizerFast, BloomModel

model_id = "meta-llama/Meta-Llama-3-8B"
# model_id = "igscience/bloom-1b7"


tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b7", force_download=True)
model = BloomModel.from_pretrained("bigscience/bloom", force_download=True)

tokenizer.save_pretrained("./tokenizers/bloom")
model.save_pretrained("./models/bloom")
