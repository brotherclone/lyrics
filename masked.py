import collections
import math
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator, \
    TrainingArguments, Trainer, get_scheduler, pipeline
from prep import load_dataset_local
from torch.optim import AdamW
from tqdm.auto import tqdm


CHUNK_SIZE = 128
PROBABILITY = 0.15
BATCH_SIZE = 64
OUTPUT_DIR = "/Users/gabrielwalsh/Sites/lyrics/models"


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(i)
        mask = np.random.binomial(1, PROBABILITY, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels
    return default_data_collator(features)


def insert_random_mask(mask_batch):
    features = [dict(zip(mask_batch, t)) for t in zip(*mask_batch.values())]
    masked_inputs = data_collator(features)
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


def group_texts(examples):
    concat_examples = {k: sum(examples[k], []) for k in examples.keys()}
    the_total_length = len(concat_examples[list(examples.keys())[0]])
    the_total_length = (the_total_length // CHUNK_SIZE) * CHUNK_SIZE
    result = {
        k: [t[i: i + CHUNK_SIZE] for i in range(0, the_total_length, CHUNK_SIZE)]
        for k, t in concat_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def get_samples():
    the_sample = lyric_data["train"].shuffle(seed=42).select(range(3))
    for i in range(3):
        print(f"LYRIC: {the_sample['lyrics'][i]}\n")
        print(f"TITLE: {the_sample['title'][i]}\n")
    return the_sample


def tokenize_samples(examples):
    result = tokenizer(examples["lyrics"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
text = "I know that you had been seen, last year on the five o'clock [MASK]."
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
lyric_data = load_dataset_local("/Users/gabrielwalsh/Sites/lyrics/data/song_data")
col_samples = get_samples()
tokenized_datasets = lyric_data.map(
    tokenize_samples,
    batched=True,
    batch_size=4,
    remove_columns=["lyrics", "title", "id", "album_id", "song_order", "trt", "notes"]
)
tokenized_samples = tokenized_datasets["train"][:3]
for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> Review {idx} length: {len(sample)}'")
concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")
chunks = {
    k: [t[i: i + CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)]
    for k, t in concatenated_examples.items()
}
for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")
for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)
for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
train_size = 70
test_size = int(0.1 * train_size)
down_sampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
logging_steps = len(down_sampled_dataset["train"]) // BATCH_SIZE
model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    output_dir=f"{model_name}-fine_tuned_lyrics",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    push_to_hub=False,
    fp16=False,
    logging_steps=logging_steps,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=down_sampled_dataset["train"],
    eval_dataset=down_sampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.train()
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
down_sampled_dataset_with_random_mask = down_sampled_dataset.remove_columns(["word_ids"])
eval_dataset = down_sampled_dataset_with_random_mask["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=down_sampled_dataset_with_random_mask["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
train_dataloader = DataLoader(
    down_sampled_dataset_with_random_mask["train"],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=BATCH_SIZE, collate_fn=default_data_collator
)
optimizer = AdamW(model.parameters(), lr=5e-5)
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_train_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(BATCH_SIZE)))
    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(OUTPUT_DIR, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(OUTPUT_DIR)

mask_filler = pipeline(
    "fill-mask", model="/Users/gabrielwalsh/Sites/lyrics/models"
)
preds = mask_filler(text)
for pred in preds:
    print(f">>> {pred['sequence']}")