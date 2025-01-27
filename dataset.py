import torch
import torch.nn as nn 
from torch.utils.data  import Dataset

class BilingualDataset(Dataset):
    def __init__(self , ds , tokenizer_src , tokinezer_tgt , src_lang , tgt_lang , seq_len ):
        super().__init__()
        self.ds = ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokinezer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        # Why subtract 2 for encoder?
        # [SOS] and [EOS] tokens are added to the sequence.
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2
        # Why subtract 1 for decoder?
        # Only [SOS] is added to the input of the decoder (the [EOS] is handled in the label).
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(" Sentence is to long ")
        

        # concat the sos the text and the eos  and pad 
        # The encoder input is constructed as:
        # [SOS] + Source Tokens + [EOS] + Padding Tokens
        # Example:
        # Input tokens: [10, 5, 7, 22]
        # Encoder input: [SOS, 10, 5, 7, 22, EOS, PAD, PAD, ...]
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens , dtype=torch.int64),
                self.eos_token,
                torch.tensor( [self.pad_token.item()] * enc_num_padding_tokens , dtype=torch.int64)
            ]
        )

        ## add sos to the deoder 

        #         The decoder input is constructed as:
        # [SOS] + Target Tokens + Padding Tokens
        # Example:
        # Target tokens: [15, 8, 19, 30]
        # Decoder input: [SOS, 15, 8, 19, 30, PAD, PAD, ...]

        dec_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens , dtype=torch.int64),
            torch.tensor([self.pad_token.item()]* dec_num_padding_tokens , dtype=torch.int64)
        ])

        ## add sos to the label the output of the decoder 
        label = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens , dtype=torch.int64),
            torch.tensor([self.pad_token]* dec_num_padding_tokens , dtype=torch.int64)
        ])
        assert encoder_input.size(0)==self.seq_len
        assert dec_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len
        # Encoder Mask (enc_mask):
        # Prevents attention to padding tokens in the encoder.
        # Shape: (1, 1, seq_len).
        # Decoder Mask (dec_mask):
        # Combines:
        # Padding mask (to avoid attending to padding tokens).
        # Causal mask (to ensure tokens can only attend to previous tokens or themselves).
        # Shape: (1, seq_len, seq_len).
        return {
            "encoder_input":encoder_input, # (seq_len)
            "dec_input":dec_input, # (seq_len)
            # all the word are pading are ok while not padding not ok 
            "encd_mask":(encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # to have (1,1,seq_len) 
            # each word can only look at the prevouis word and each can only loock not paddding words and not to see the word after it 
            "dec_mask":(dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & cusal_mask(dec_input.size(0)), # so we have (1,seq) & (1,swq , seq ) so we get (1,seq , seq ) plan of true of false 
            "label":label ,# (seq),
            "src_text": src_text,
            "tgt_text":tgt_text
        }
def cusal_mask(size):
    mask =torch.triu(torch.ones(1 , size , size ) , diagonal=1).type(torch.int)
    return mask ==0
    
