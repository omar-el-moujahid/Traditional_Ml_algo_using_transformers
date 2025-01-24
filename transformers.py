#!/usr/bin/env python
# coding: utf-8

# In[56]:


import torch
import torch.nn as nn


# In[57]:


import math 


# In[58]:


class Inputembedding(nn.Module):
    def __init__(self,d_model:int , vocab_size : int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedkding = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)


# In[59]:


class PositionalEncodimg(nn.Module):
    def __init__(self, d_model : int , seq_len : int , dropout : float ):
        super().__init__()
        self.d_model=d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) 
        
        ## create a matrix of shape (seq , d_model )
        pe= torch.zeros(seq_len, d_model)
        # create vector of shape (seq_len )
        position=torch.arange(0,seq_len,dtype=torch.float).unique(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        # applay the sin to even position 
        pe[:,0::2]=torch.sin(position*div_term)
        # applay thr cos   to the odd postion 
        pe[:,1::2]=torch.cos(position*div_term)

        pe = pe.unsqueeze(0) ##(1 , deq_len , d_model)
        self.register_buffer('pe',pe)
    def forward(self , x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


# In[60]:


class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10**-6 )-> None :
        super().__init__()
        self.eps=eps
        self.bias=nn.Parameter(torch.zeros(1))   # for additional 
        self.alpha=nn.Parameter(torch.ones(1)) # for multiplication
    
    def forward(self , x):
        mean = x.mean(dim=-1,keepdim=True )
        std = x.std(dim=-1,keepdim=True )
        return self.alpha*(x-mean) / (std + self.eps) + self.bias


# In[61]:


## FFN(x) = (x.W1 + b1 )*W2 + b2 W1(d_model,d_ff) and W2(d_ff , d_model )
# we use it after the attention it s a fully connected feed forword network
# applay for each position 
# d_model: The dimensionality of the input and output (e.g., 512 or 768 in many architectures).
#d_ff: The dimensionality of the hidden layer. Typically much larger than d_model (e.g., 2048 in many architectures), allowing the model to learn richer feature transformations.
#dropout: A regularization technique to prevent overfitting by randomly dropping some connections during training
class  Feed_Forword(nn.Module):
    def __init__(self, d_model : int , d_ff : int  , dropout :float):
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff) # for W1 b1 
        self.dropout = nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model) # for W2 b2
    def forward(self , x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# In[62]:


class MultiheadAttenstion(nn.Module):
    def __init__(self, d_model : int ,h :int , dropout : float ):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h==0,"d_model is not devede by h   "
        self.d_k = d_model // h
         # Learnable linear projections for queries, keys, values, and output
        self.w_q = nn.Linear(d_model, d_model)  # For projecting queries
        self.w_k = nn.Linear(d_model, d_model)  # For projecting keys
        self.w_v = nn.Linear(d_model, d_model)  # For projecting values
        self.w_0 = nn.Linear(d_model, d_model)  # Final output projection
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def calculat_attenstion( q , k , v , mask , dropout : nn.Dropout ):
        attemstion_score= ((q@k.transpose(-2,-1)) / math.sqrt(q.shape[-1]))
        if mask is not None:
            attemstion_score.masked_fill_(mask==0,-100)
        attemstion_score = attemstion_score.softmax(dim =-1)
        if dropout is not None:
            attemstion_score=dropout(attemstion_score )
        return (attemstion_score @ v) , attemstion_score
    def forward(self, q,k,v,mask ):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # we are giong from (batch , seq , d_model ) to firs-> (batch , seq , h , d_k)
        #  -> and then using transpose to switch between the secodes demension and 
        # the third so we get (batch , h , seq , d_k)
        # the porpose to have (seq , d_k)
        # so we did it for all the three
        query=query.view(query.shape[0],query.shape[1], self.h , self.d_k).transpose(1,2)
        key=key.view(key.key[0],key.shape[1], self.h , self.d_k).transpose(1,2)
        value=value.view(value.value[0],value.shape[1], self.h , self.d_k).transpose(1,2)
        x , attemstion_score = MultiheadAttenstion.calculat_attenstion(query,key,value,mask , self.dropout)
        ## concat the heads now  
        # first we trnaspose them to have the initale spreat (batch , seq , h , d_k)
        x=x.transpose(1,2)
        # now we should concat the h and d_k to have d_module
        # Merge heads into d_model
        x = x.contiguous().view(x.shape[0], x.shape[1], self.h*self.d_k) 
        # final target is 
        x= self.w_0(x)


# In[63]:


class ResidualCongitnection(nn.Module):
    def __init__(self, dropout : float):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
    def forward(self, x,sublayer):
        ## means that sublayer witch could be  represent multihead_attention_layer or
        #  what ever gonna have the same Normalization as x 
        return x+self.dropout(sublayer(self.norm(x)))


# In[72]:


class EncoderBlock(nn.Module):
    def __init__(self, self_attentio : MultiheadAttenstion , feed_forword : Feed_Forword , dropout : float):
        super().__init__()
        self. self_attentio = self_attentio 
        self.feed_forword = feed_forword
        self.connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self , x , src_masck):
        x=self.connection[0](x, lambda x : self.self_attentio(x,x,x,src_masck))
        x=self.connection[1](x, self.feed_forword)
        return x 


# In[65]:


class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm = LayerNormalization()
    def forward(self , x, mask):
        for layer in self.layers:
            x=layer(x , mask)
        return self.norm(x)


# In[66]:


class DencoderBlock(nn.Module):
    def __init__(self, self_attentio : MultiheadAttenstion , feed_forword : Feed_Forword , dropout : float):
        super().__init__()
        self. self_attentio = self_attentio 
        self.feed_forword = feed_forword
        self.connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    def forward(self , x  , k  , v, src_masck , tgt_mask):
        x=self.connection[0](x, lambda x : self.self_attentio(x,x,x,tgt_mask))
        # result from the encoder  the k and the value 
        x=self.connection[1](x, lambda x : self.self_attentio(x,k,v,src_masck))
        x=self.connection[2](x, self.feed_forword)
        return x


# In[67]:


class Dencoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm = LayerNormalization()
    def forward(self , x  , k  , v, src_masck , tgt_mask):
        for layer in self.layers:
            x=layer(x  , k  , v, src_masck , tgt_mask)
        return self.norm(x)


# In[68]:


class ProjectionLayer(nn.Module):
    def __init__(self, d_model , vocab_size):
        super().__init__()
        self.linear=nn.Linear(d_model, vocab_size)
    def forward(self , x ):
        # (batch , seq_len , d_model) -> (batch , seq_len , vocab_size)
        return torch.log_softmax(self.linear(x) , dim=-1)


# In[69]:


class Transformer(nn.Module):
    def __init__(self, encoder : Encoder , decoder : Dencoder ,srcEmbedding  : Inputembedding ,targettembedding   : Inputembedding , src_position : PositionalEncodimg , target_position : PositionalEncodimg, projection = ProjectionLayer ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.srcEmbedding = srcEmbedding
        self.targettembedding = targettembedding
        self.src_position = src_position
        self.target_position = target_position
        self.projection=projection
    def encoder(self , src , src_mask ):
        src = self.srcEmbedding(src)
        src = self.src_position(src)
        return self.encoder(src , src_mask)
    def decode(self , tgt , tgt_mask):
        tgt = self.targettembedding(tgt)
        tgt = self.target_position(tgt)
        return self.encoder(tgt , tgt_mask)
    def project(self , x):
        return self.project(x)


# In[71]:


def h( src_vocab_size : int , tgt_vocab_size : int , src_seq_len : int , tgt_seq_len : int , d_model :int = 512 , n : int = 6  ,  h : int  = 8  , dropout : int =0.1 , d_ff : int = 2048 ) -> Transformer:
    
    src_embed = Inputembedding(d_model , src_vocab_size)
    tgt_emdeb = Inputembedding(d_model , tgt_vocab_size)

    src_position = PositionalEncodimg(d_model,src_seq_len , dropout)
    tgt_position = PositionalEncodimg(d_model,tgt_seq_len , dropout)

    # creat n encoder 

    encoder_bloks=[]
    for _ in  range(n):
        encoder_self_attention = MultiheadAttenstion(d_model,h,dropout)
        feed_Forword = Feed_Forword(d_model,d_ff,dropout)
        encoder_blok = EncoderBlock(encoder_self_attention,feed_Forword,dropout)
        encoder_bloks.append(encoder_blok)
    
    # decoder 
    dencoder_bloks=[]
    for _ in  range(n):
        encoder_self_attention = MultiheadAttenstion(d_model,h,dropout)
        feed_Forword = Feed_Forword(d_model,d_ff,dropout)
        dencoder_blok = DencoderBlock(encoder_self_attention,feed_Forword,dropout)
        dencoder_bloks.append(dencoder_blok)
    
    encoder = Encoder(nn.ModuleList(encoder_bloks) )
    decoder = Dencoder(nn.ModuleList(dencoder_bloks))
    # projection layer 
    project = ProjectionLayer(d_model , tgt_vocab_size) 
    # tamsformer 
    transformer = Transformer(encoder,decoder,src_embed,tgt_emdeb,src_position,tgt_position,project)
    # initia the parammeters 
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return transformer


# In[ ]:




