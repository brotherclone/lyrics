from datasets import load_from_disk


def load_dataset_local(dataset_path):
    ds = load_from_disk(dataset_path)
    return ds


if __name__ == '__main__':
    d = load_dataset_local("/Users/gabrielwalsh/Sites/lyrics/data")
    split_set = d.train_test_split(train_size=0.8, seed=41)
    dataset_clean = split_set["train"].train_test_split(train_size=0.8, seed=42)
    dataset_clean["validation"] = dataset_clean.pop("test")
    dataset_clean['test'] = split_set['test']
    dataset_clean.save_to_disk("/Users/gabrielwalsh/Sites/lyrics/data/song_data")