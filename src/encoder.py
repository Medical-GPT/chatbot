from constants import ENCODER_TOKENS, ENCODER_ENCTEXT
import gc
import numpy as np

class Encoder:

    def __init__(self, path, load_tokens=False, encode_text=True):

        if load_tokens:
            self.tokens = self.load_tokens()
        else:
            data = self.read_data(path)
            self.tokens = self.generate_tokens(data)
            del data
            gc.collect()
            self.save_tokens(self.tokens)

        self.encode, self.decode = self.generate_coders(self.tokens)

        if encode_text:
            self.encode_and_save(path)

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        print("Loaded data")
        return data

    def generate_tokens(self, data):
        return sorted(list(set(data)))

    def save_tokens(self, tokens):
        with open(ENCODER_TOKENS, "w", encoding="utf-8") as f:
            f.write("".join(tokens))

        print(f"Saved {len(tokens)} tokens to {ENCODER_TOKENS}")
        print(tokens)

    def load_tokens(self):
        with open(ENCODER_TOKENS, "r", encoding="utf-8") as f:
            tokens = f.read()
        return tokens

    def get_vocab_size(self):
        return len(self.tokens)

    def generate_coders(self, tokens):
        stoi = {ch: i for i, ch in enumerate(tokens)}
        itos = {i: ch for i, ch in enumerate(tokens)}
        encode = lambda s: [
            stoi[c] for c in s
        ]  # encoder: take a list of tokens, output a list of integers
        decode = lambda l: "".join(
            [itos[i] for i in l]
        )  # decoder: take a list of integers, output a string

        return encode, decode

    def encode_and_save(self, path):
        with open(path) as input_file:
            with open(ENCODER_ENCTEXT, "wb") as output_file:
                for line in input_file:
                    # Example encoded line of integers
                    encoded_line = self.encode(line+"\n")

                    # Convert to numpy array of uint8 (8-bit unsigned integers)
                    encoded_line = np.array(encoded_line, dtype=np.uint8)

                    # Save encoded line to binary file
                    output_file.write(encoded_line.tobytes())
        print(f"Saved encoded text to {ENCODER_ENCTEXT}")