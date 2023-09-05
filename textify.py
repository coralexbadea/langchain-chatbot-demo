from datasets import load_dataset
from unidecode import unidecode

train_txt_path = "train.txt"

dataset = load_dataset("coralexbadea/monitorul_trial")

train_dataset1 = dataset["train"]["text"][:10]
combined_train_dataset = train_dataset1

def generate_txt(txt_path, dataset):
    with open(txt_path, 'w', newline='') as text_file:
        for case in dataset:
            text = unidecode(case)
            text_file.write(text + "\n")
            
            
generate_txt(train_txt_path, combined_train_dataset)