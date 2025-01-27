import torch 
import torch.nn as nn 
from datasets import load_dataset
from torch.utils.data import Dataset , DataLoader , random_split
## tokenizer to spli the sentenses into many word using many type we gonna work wiyh 
# the WordLevel and map then into numbers the number of the word im the vocabilure
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from tokenizers import Tokenizer
## tyoe of tokinezer 
from tokenizers.models import WordLevel
# trin the tokinezer base on the sentences given 
from tokenizers.trainers import WordLevelTrainer
# split the word using spaces 
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilingualDataset , cusal_mask

import warnings

from importnb import Notebook

from torch.utils.tensorboard import SummaryWriter
# Import the `transformers.ipynb` file
from transformer import build_transformer

from config import get_config , get_weights_file_path

from tqdm import tqdm
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
        special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        trainer = WordLevelTrainer(
            special_tokens=special_tokens,
            min_frequency=2
        )
        # ??
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer)
        
        # Verify special tokens are in the vocabulary
        vocab = tokenizer.get_vocab()
        for token in special_tokens:
            if token not in vocab:
                raise ValueError(f"Special token {token} not in vocabulary after training")
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
    # ds_rows = load_dataset('Helsinki-NLP/opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    dataset_name = 'Helsinki-NLP/opus_books'
    language_pair = f"{config['lang_src']}-{config['lang_tgt']}"
    print(f"Loading dataset {dataset_name} for language pair {language_pair}...")
      # Use load_dataset with error handling
    try:
        ds_rows = load_dataset(
            dataset_name,
            language_pair,
            split='train',
            cache_dir='./dataset_cache'
        )
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        # Fallback to direct loading
        ds_rows = load_dataset(
            'opus_books',
            language_pair,
            split='train',
            cache_dir='./dataset_cache'
        )
    
    print(f"Successfully loaded {len(ds_rows)} translation pairs")
    # build tokenizers for each lang 
    tokenizer_src = get_or_build_tokenizer(config , ds_rows , config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config , ds_rows , config["lang_tgt"])

    # split the data into validation and tain 
    train_ds_size = int(0.9 * len(ds_rows))
    val_ds_size =  len(ds_rows) - train_ds_size
    train_ds , val_ds = random_split(ds_rows , [train_ds_size,val_ds_size])
    train_ds = BilingualDataset(train_ds,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    val_ds = BilingualDataset(val_ds,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    max_len_src=0
    max_len_tgt=0
    for item in ds_rows:
        src_ids = len(tokenizer_src.encode(item['translation'][config["lang_src"]]).ids)
        tgt_ids = len(tokenizer_tgt.encode(item['translation'][config["lang_tgt"]]).ids)
        max_len_src = max(max_len_src, src_ids)
        max_len_tgt = max(max_len_tgt, tgt_ids)
    print(f'max lenght in the src langague is {max_len_src}')
    print(f'max lenght in the tgt langague is {max_len_tgt}')
    train_dataloader = DataLoader(train_ds ,batch_size=config["batch_size"] , shuffle=True)
    val_dataloader =DataLoader(val_ds ,batch_size=config["batch_size"] , shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
def get_model (config , src_vocab_size ,tgt_vocab_size ):
    return build_transformer(src_vocab_size , tgt_vocab_size , config["src_seq_len"] , config["tgt_seq_len"] , config["d_model"])

def train_modal(config):
    device = torch.device('code' if torch.cuda.is_available() else "cpu")
    print(f"im using {device}")

    ## check if folder existe for  weights 
    Path(config["model_folder"]).mkdir(parents=True,exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config=config) 

    model= get_model(config ,tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size() ).to(device)

    # tensorboard visual the graphes
    write = SummaryWriter(config["experiment_name"])
    # optimizer the moxel using adams
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"],eps=1e-9)

    ## in case the model cruch we restore the state of the model and the sate of the optimizer 
    initial_epoch =0 
    global_step=0
    if (config["preload"]):
        model_filename = get_weights_file_path(config,initial_epoch)
        print(f"preloading the model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch =state["epoch"]+1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    ## using cross entrepy loss we want it to ignore the pading and not effecting in the 
    # loss functiom and label_smoothing to true for alowsing us to be low confident 
    # for dession to not be avoiding overrfiting 
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=True)

    for epoch in range(initial_epoch , config['num_epochs']):
        model.train()
        ## batch itorator for what ??
        batch_itorator=tqdm(train_dataloader , desc=f'processing epoch {epoch}')

        for batch in batch_itorator:
            encoder_input = batch['encoder_input'].to(device) # (batch ,seq_len)
            decoder_input = batch['dec_input'].to(device) # (batch ,seq_len)
            encoder_mask=batch["encd_mask"].to(device) #(bach , 1,1,seq_len)
            decoder_mask=batch["dec_mask"].to(device) #(bach  , 1,1,seq_len) hide only embadding
            
            # Run the tensors through the transformers 

            encoder_output = model.encode(encoder_input,encoder_mask) # (batch , seq , d_model )
            decoder_output = model.decode(encoder_output , encoder_mask,decoder_input,decoder_mask) # (batch , seq_len , d_model )
            projection = model.project(decoder_output) # (batch , seq_len , tgt_vocab_zize )

            label = batch["label"].to(device) #(batch , seq)

            #(0,seq_len , tgt_vocab_zize ) --> ( bathc * seq_len , tgt_vocab_size)
            loss = loss_fn(projection.view(-1,tokenizer_tgt.get_vocab_size()) , label.view(-1))
            batch_itorator.set_postfix({f"loss":f"{loss.item():6.3f}"})

            # log the loss
            write.add_scalar('tain loss',loss.item(),global_step)
            write.flush()

            # backpropagation the loss
            loss.backward()

            # uppdate the weights 
            optimizer.step()
            optimizer.zero_grad()

            global_step +=1

            ## save the model at he end of evry epoch 
            model_filename=get_weights_file_path(config,f'{epoch:02d}')
            torch.save(
                {
                    "epoch":epoch,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "global_step":global_step
                } , model_filename
            )
if __name__ == "__main__":
    # warnings.filterwarnings('ignore')
    confg = get_config()
    train_modal(confg)