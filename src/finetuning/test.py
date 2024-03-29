from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys
import torch

# Get the path to the finetuned model
model_path = sys.argv[1]

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Add this line
# Set the model to evaluation mode
model.eval()

while True:
    prompt = input("Input: ")
    # Encode the prompt using the tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Generate text using the model
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=150,
        do_sample=True,
        top_k=50,
        temperature=0.6,
    )

    # Decode the generated text and print it
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Response: {generated_text}")
