
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, max_length=100):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example
if __name__ == "__main__":
    prompt = "Artificial Intelligence is revolutionizing"
    output = generate_text(prompt)
    print("\nGenerated Text:\n")
    print(output)

