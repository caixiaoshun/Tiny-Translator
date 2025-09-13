import csv
from tokenizers import models, Tokenizer, normalizers, pre_tokenizers, decoders, trainers, processors


def text_iterator(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['0'] + " " + row['1']
            yield text

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC()
])

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=30_000,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
)

tokenizer.train_from_iterator(text_iterator("data/wmt_zh_en_training_corpus.csv"), trainer=trainer)



tokenizer.post_processor = processors.TemplateProcessing(
    single="[SOS] $A [EOS]",
    pair="[SOS] $A [EOS] $B [EOS]",
    special_tokens=[
        ("[SOS]", tokenizer.token_to_id("[SOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ],
)

tokenizer.save("checkpoints/tokenizer.json")