"""using the huggingface tokenizers module"""
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.pre_tokenizers import Whitespace


def learn_tokenizer(data_file_name, vocab_size=100000, min_frequency=2, init_alphabet=None):
    if init_alphabet is None:
        init_alphabet = pre_tokenizers.ByteLevel.alphabet()

    # init tokenizer model
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = Whitespace()  # pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    # tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # specify setting and train the tokenizer on the given text
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=init_alphabet
    )
    tokenizer.train([
        data_file_name + ".txt",
    ], trainer=trainer)

    # save trained tokenizer
    tokenizer.save(data_file_name + "_bpe.tokenizer.json", pretty=True)


def load_tokenizer(data_file_name):
    tokenizer = Tokenizer.from_file(data_file_name + "_bpe.tokenizer.json")
    return tokenizer


# if __name__ == "__main__":
#     file_name = 'shakespeare_char\\input'
#     learn_tokenizer(file_name)
#     tokenizer = load_tokenizer(file_name)
#     enc = tokenizer.encode('Hello everyone this is a example text.')
#     dec = tokenizer.decode(enc.ids)
#     tokenizer.get_vocab()
#     pass

