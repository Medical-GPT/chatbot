from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys

# Get the path to the finetuned model
path = sys.argv[1]

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained(path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Add this line
# Set the model to evaluation mode
model.eval()
prompt = "Patient: Hi doctor, I've been experiencing a lot of chest pain and shortness of breath lately, even when I'm not doing any physical activity.\nDoctor:"

while True:
    # Define a prompt for the model to complete

    # Encode the prompt using the tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using the model
    output = model.generate(
        input_ids=input_ids, max_length=100, do_sample=True, top_k=50, temperature=0.05
    )

    # Decode the generated text and print it
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

    text = input("Input: ")
    prompt = f"Patient: {text}\nDoctor:"
