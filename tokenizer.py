import os
import time
from tqdm import tqdm
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors

# Define paths
output_dir = "/kaggle/working"
tokenizer_path = os.path.join(output_dir, "custom_tokenizer.json")
merges_path = os.path.join(output_dir, "merges.txt")
vocab_path = os.path.join(output_dir, "vocab.txt")

files = [
    "/kaggle/input/openwebtext-dataset/train_split.txt",
    "/kaggle/input/openwebtext-dataset/val_split.txt",
]

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# Lowercase everything and strip leading/trailing whitespace.
tokenizer.normalizer = normalizers.Sequence([
    normalizers.Lowercase(),
    normalizers.Strip()
])

# Use ByteLevel pre-tokenizer but disable the automatic addition of a prefix space.
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
special_tokens = [
    "[UNK]",    # Unknown token
    "<s>",      # BOS token (id=1)
    "</s>",     # EOS token (id=2)
    "<|prompter|>",
    "<|assistant|>"
]

"""This tokenizer is designed to handle special tokens and is configured to work with the BPE model.
It includes a normalizer to lowercase text and strip whitespace, a ByteLevel pre-tokenizer to handle byte-level tokenization, and a decoder to convert tokens back to text. The tokenizer is trained using a BPE trainer with a specified vocabulary size and special tokens."""


trainer = trainers.BpeTrainer(
    vocab_size=32173,
    special_tokens=special_tokens,
)

def count_lines(file_paths):
    total_lines = 0
    for file in file_paths:
        with open(file, "r", encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
    return total_lines

total_lines = count_lines(files)
print(f"Total lines to process: {total_lines}")

print("\n>>>> processing text for training...")
with tqdm(total=total_lines, desc="Processing text", unit=" lines") as pbar:
    def get_iterator():
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()
                    pbar.update(1)

    start_time = time.time()
    tokenizer.train_from_iterator(get_iterator(), trainer=trainer)
print(f">>>> Text processing complete in {time.time() - start_time:.2f} seconds.")

print("\n>>>> Saving tokenizer JSON file...")
start_time = time.time()
tokenizer.save(tokenizer_path)
print(f">>>> Tokenizer JSON saved to: {tokenizer_path} in {time.time()-start_time:.2f} seconds.")

print("\n>>>> Extracting merges.txt...")
bpe_model = tokenizer.model
if isinstance(bpe_model, models.BPE):
    start_time = time.time()
    try:
        with open(merges_path, "w", encoding="utf-8") as f:
            for merge in tqdm(bpe_model.get_merges(), desc="Saving merges", unit=" merges"):
                f.write(" ".join(merge) + "\n")
        print(f">>>> Merges file saved to: {merges_path} in {time.time()-start_time:.2f} seconds.")
    except Exception as e:
        print("*** Error saving merges.txt:", e)


print("\n>>>> Extracting vocab.txt...")
if isinstance(bpe_model, models.BPE):
    start_time = time.time()
    try:
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token, _ in tqdm(bpe_model.get_vocab().items(), desc="Saving vocab", unit=" tokens"):
                f.write(token + "\n")
        print(f"✅ Vocabulary file saved to: {vocab_path} in {time.time()-start_time:.2f} seconds.")
    except Exception as e:
        print("*** Error saving vocab.txt:", e)

print("\n🎉 All processes completed successfully!")
