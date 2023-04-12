from .constants import ENCODER_TOKENS, ENCODER_ENCTEXT
import gc
import numpy as np


class Encoder:
    def __init__(self, path, load_tokens=False, encode_text=False):
        if load_tokens:
            self.tokens = Encoder.load_tokens()
        else:
            data = self.read_data(path)
            self.tokens = self.generate_tokens(data)
            del data
            gc.collect()
            Encoder.save_tokens(self.tokens)

        self.encode, self.decode = Encoder.generate_coders(self.tokens)

        if encode_text:
            self.encode_and_save(path)

    @staticmethod
    def save_tokens(tokens, dest=ENCODER_TOKENS):
        with open(dest, "w", encoding="utf-8") as f:
            f.write("".join(tokens))

        print(f"Saved {len(tokens)} tokens to {dest}")
        print(tokens)

    @staticmethod
    def load_tokens(path=ENCODER_TOKENS):
        with open(path, "r", encoding="utf-8") as f:
            tokens = f.read()
        return tokens

    @staticmethod
    def generate_coders(tokens):
        stoi = {ch: i for i, ch in enumerate(tokens)}
        itos = {i: ch for i, ch in enumerate(tokens)}
        encode = lambda s: [
            stoi[c] for c in s
        ]  # encoder: take a list of tokens, output a list of integers
        decode = lambda l: "".join(
            [itos[i] for i in l]
        )  # decoder: take a list of integers, output a string

        return encode, decode

    @staticmethod
    def generate_coders_from_path(path):
        tokens = Encoder.load_tokens(path)
        return Encoder.generate_coders(tokens)

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        print("Loaded data")
        return data

    def generate_tokens(self, data):
        return sorted(list(set(data)))

    def get_vocab_size(self):
        return len(self.tokens)

    def get_tokens(self):
        return self.tokens

    def encode_and_save(self, path):
        with open(path) as input_file:
            with open(ENCODER_ENCTEXT, "wb") as output_file:
                for line in input_file:
                    # Example encoded line of integers
                    encoded_line = self.encode(line + "\n")

                    # Convert to numpy array of uint8 (8-bit unsigned integers)
                    encoded_line = np.array(encoded_line, dtype=np.uint8)

                    # Save encoded line to binary file
                    output_file.write(encoded_line.tobytes())
        print(f"Saved encoded text to {ENCODER_ENCTEXT}")
