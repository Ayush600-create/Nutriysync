from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/DialoGPT-small"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("✅ Model and tokenizer loaded successfully!")
