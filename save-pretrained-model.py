from transformers import BloomTokenizerFast, BloomModel

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom", force_download=True)
model = BloomModel.from_pretrained("bigscience/bloom", force_download=True)

tokenizer.save_pretrained("./tokenizers/bloom")
model.save_pretrained("./models/bloom")
