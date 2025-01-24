import torch 
import torch.nn as nn 
from datasets import load_dataset
from torch.utils.data import Dataset , DataLoader , random_split
## tokenizer to spli the sentenses into many word using many type we gonna work wiyh 
# the WordLevel and map then into numbers the number of the word im the vocabilure

from tokenizers import Tokenizer
## tyoe of tokinezer 
from tokenizers.models import WordLevel
# trin the tokinezer base on the sentences given 
from tokenizers.trainers import WordLevelTrainer
# split the word using spaces 
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilingualDataset , cusal_mask

from importnb import Notebook

# Import the `transformers.ipynb` file
with Notebook():
    from transformers import build_transformer


## buil the tokinezer
def get_or_build_tokenizer(config , ds , lang):
    tokinezer_path = Path(config["tokinezer_file"].format(lang))
    # if it not found we creat the tokenization 
    if not Path.exists(tokinezer_path):
        # initial the tokenizer by WordLevel method/model 
        tokenizer = Tokenizer(WordLevel(unk_token="UNK"))
        # split it using Whitespace
        tokenizer.pre_tokenizer=Whitespace()
        # train the split the sentence should have at least two of the splecial tokens 
        # [UNK]: Unknown token
        # [PAD]: Padding token for sequences
        # [SOS]: Start-of-sequence token
        # [EOS]: End-of-sequence token
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"] , min_frequency=2)
        # ??
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer)
        ## save the tokenzation
        tokenizer.save(str(tokinezer_path))
    # if it existe we use it derectely 
    else:
        tokenizer= Tokenizer.from_file(str(tokinezer_path))
    return tokenizer

def get_all_sentences(ds,lang):
    for items in ds:
        yield items['translation'][lang]

def get_dataset(config):
    ds_rows = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    # build tokenizers for each lang 
    tokenizer_src = get_or_build_tokenizer(config , ds_rows , config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config , ds_rows , config["lang_tgt"])

    # split the data into validation and tain 
    train_ds_size = int(0.9 * len(ds_rows))
    val_ds_size = int(0.1 * len(ds_rows))
    train_ds , val_ds = random_split(ds_rows , [train_ds_size,val_ds_size])
    train_ds = BilingualDataset(train_ds,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    val_ds = BilingualDataset(val_ds,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    max_len_src=0
    max_len_tgt=0
    for item in ds_rows:
        max_len_src=max(max_len_src, tokenizer_src.encode(item["translation"][config["lang_src"]]))
        max_len_tgt=max(max_len_tgt, tokenizer_src.encode(item["translation"][config["lang_tgt"]]))
    print(f'max lenght in the src langague is {max_len_src}')
    print(f'max lenght in the tgt langague is {max_len_tgt}')
    train_dataloader = DataLoader(train_ds ,batch_size=config["batch_size"] , shuffle=True)
    val_dataloader =DataLoader(val_ds ,batch_size=config["batch_size"] , shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
def get_model (config , src_vocab_size ,tgt_vocab_size ):
    return build_transformer( src_vocab_size , tgt_vocab_size , config["src_seq_len"] , config["tgt_seq_len"]  )
