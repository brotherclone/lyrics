import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from prep import load_dataset_local

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def cls_pooling(the_model_output):
    return the_model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    device = torch.device("cpu")
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


if __name__ == '__main__':
    test_data = load_dataset_local("/Users/gabrielwalsh/Sites/lyrics/data")
    model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    embedding = get_embeddings(test_data["lyrics"])
    embeddings_dataset = test_data.map(
        lambda x: {"embeddings": get_embeddings(x["lyrics"]).detach().cpu().numpy()[0]}
    )
    embeddings_dataset.add_faiss_index(column="embeddings")
    lyric = "There's Mister Sunshine, he's at it again."
    lyric_embedding = get_embeddings([lyric]).cpu().detach().numpy()
    scores, samples = embeddings_dataset.get_nearest_examples(
        "embeddings", lyric_embedding, k=5
    )
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)
    for _, row in samples_df.iterrows():
        print(f"SCORE: {row.scores}")
        print(f"TITLE: {row.title}")
