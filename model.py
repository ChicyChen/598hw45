import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from mamba import MambaBlock, Mamba, MambaConfig
from rotary_embedding_torch import RotaryEmbedding


class AR(Dataset):
    # In induction head we have ngram = 1. But the code provided is for general ngram setting. While using this, initialize ngram = 1. 
    """ Naive associative recall dataset """
    def __init__(self, num_examples, tokenizer, n_gram=1, M=1, n_ctx = 1024, seed = 0, train_split=0.8):
        self.num_examples = num_examples
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.seed = seed
        self.n_gram = n_gram
        self.M = M
        x, y = self.data_gen()
        if train_split:
            self.train_x, self.train_y, self.test_x, self.test_y = self.split(x, y, train_split)
            self.train = self.numpy_to_tensor_dataset(self.train_x, self.train_y)
            self.test = self.numpy_to_tensor_dataset(self.test_x, self.test_y)
        else:
            self.train_x, self.train_y, self.test_x, self.test_y = x, y, None, None
            self.train = self.numpy_to_tensor_dataset(self.train_x, self.train_y)
            self.test = None
    def get_str_dataset(self, split="train"):
        if split == "train":
            x_str = [self.tokenizer.decode(xi) for xi in self.train_x]
            y_str = [self.tokenizer.decode([yi]) for yi in self.train_y]
        elif split == "test":
            x_str = [self.tokenizer.decode(xi) for xi in self.test_x]
            y_str = [self.tokenizer.decode([yi]) for yi in self.test_y]
        else:
            raise ValueError("split should be either 'train' or 'test'")
        return x_str, y_str
    def numpy_to_tensor_dataset(self, x, y):
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return TensorDataset(x, y)
    def gen_single_example(self):
        # get the vocab size
        def count(str_x, str_n_gram_head):
            counts = sum([
                str_x.startswith(str_n_gram_head, i) for i in range(len(str_x))
            ])
            return counts
        def gen_x():
            gen_x_success = False
            while not gen_x_success:
                x = np.random.choice(vocab, self.n_ctx-self.n_gram*2, replace=True).tolist()
                # remove the case where the n_gram_head is repeated in the sequence
                for _ in range(10):
                    pos = [i for i in range(len(x)-len(n_gram_head)+1) if x[i:i+len(n_gram_head)] == n_gram_head]
                    if len(pos) == 0:
                        gen_x_success = True
                        break
                    else:
                        # remove the n_gram_head from x
                        # get all positions of the n_gram_head in x
                        for p in reversed(pos):
                            # remove len(n_gram_head) elements from x starting from p
                            x = x[:p] + x[p+len(n_gram_head):]
                        # fill the rest of the sequence with random elements
                        x.extend(np.random.choice(vocab, self.n_ctx-self.n_gram*2-len(x), replace=True).tolist())
                x_test = " ".join([str(xi) for xi in x])
                if count(x_test, str_n_gram_head) == 0:
                    gen_x_success = True

            x_test = x + n_gram_head
            # check if there's only one n_gram_head in the sequence
            # to avoid the case where the n_gram_head has 
            # repeated structure such as x= [1, 2, 3, 1] , n_gram_head = [1, 1]
            str_x_test = " "+" ".join([str(xi) for xi in x_test])+ " "
            if count(str_x_test, str_n_gram_head) > 1:
                print("Error in gen_x")
                print(f"str_x_test: {str_x_test}", f"str_n_gram_head: {str_n_gram_head}", 
                      "count: ", count(str_x_test, str_n_gram_head))
            if count(str_x_test, str_n_gram_head) == 1:
                return x
            else:
                return None
        def insert_n_gram_head(x):
            pos = random.randint(0, len(x)-1)
            y = x[pos]
            x_new = x[:pos] + n_gram_head + x[pos:] + n_gram_head
            str_x_new = " "+" ".join([str(xi) for xi in x_new])+" "

            if count(str_x_new, str_n_gram_head) == 2:
                return x_new, y
            else:
                return None, None
        vocab_size = len(self.tokenizer)
        vocab = list(range(vocab_size))
        # set a deterministic n_gram_head
        n_gram_head = list(range(self.n_gram))
        switch_head = np.random.choice(self.M, 1)
        n_gram_head[0] = switch_head[0]
       
        str_n_gram_head = " "+" ".join([str(xi) for xi in n_gram_head])+" "
        assert self.n_gram*2 < self.n_ctx, "n_gram*2 should be less than n_ctx"
        success = False
        while not success:
            x = gen_x()
            if x is not None:
                for _ in range(10):
                    x_new, y = insert_n_gram_head(x)
                    if x_new is not None:
                        success = True
                        break
        return x_new, y
            
    def data_gen(self):
        x = []
        y = []
        # get previous random status and recover after generating the dataset
        random_status = random.getstate()
        random.seed(self.seed)
        for i in range(self.num_examples):
            if i % 1000 == 0:
                print(f"Generating example {i}")
            xi, yi = self.gen_single_example()
            x.append(xi)
            y.append(yi)
        x = np.array(x)
        y = np.array(y)
        random.setstate(random_status)
        return x, y
    def split(self, x, y, train_ratio = 0.8):
        num_train = int(len(x)*train_ratio)
        train_x = x[:num_train]
        train_y = y[:num_train]
        test_x = x[num_train:]
        test_y = y[num_train:]
        return train_x, train_y, test_x, test_y


class InductionAR(Dataset):
    # In induction head we have ngram = 1. But the code provided is for general ngram setting. While using this, initialize ngram = 1. 
    """ Naive associative recall dataset """
    def __init__(self, num_examples, tokenizer, n_gram=1, n_ctx = 1024, seed = 0, train_split=0.8):
        self.num_examples = num_examples
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.seed = seed
        self.n_gram = n_gram
        x, y = self.data_gen()
        if train_split:
            self.train_x, self.train_y, self.test_x, self.test_y = self.split(x, y, train_split)
            self.train = self.numpy_to_tensor_dataset(self.train_x, self.train_y)
            self.test = self.numpy_to_tensor_dataset(self.test_x, self.test_y)
        else:
            self.train_x, self.train_y, self.test_x, self.test_y = x, y, None, None
            self.train = self.numpy_to_tensor_dataset(self.train_x, self.train_y)
            self.test = None
    def get_str_dataset(self, split="train"):
        if split == "train":
            x_str = [self.tokenizer.decode(xi) for xi in self.train_x]
            y_str = [self.tokenizer.decode([yi]) for yi in self.train_y]
        elif split == "test":
            x_str = [self.tokenizer.decode(xi) for xi in self.test_x]
            y_str = [self.tokenizer.decode([yi]) for yi in self.test_y]
        else:
            raise ValueError("split should be either 'train' or 'test'")
        return x_str, y_str
    def numpy_to_tensor_dataset(self, x, y):
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return TensorDataset(x, y)
    def gen_single_example(self):
        # get the vocab size
        def count(str_x, str_n_gram_head):
            counts = sum([
                str_x.startswith(str_n_gram_head, i) for i in range(len(str_x))
            ])
            return counts
        def gen_x():
            gen_x_success = False
            while not gen_x_success:
                x = np.random.choice(vocab, self.n_ctx-self.n_gram*2, replace=True).tolist()
                # remove the case where the n_gram_head is repeated in the sequence
                for _ in range(10):
                    pos = [i for i in range(len(x)-len(n_gram_head)+1) if x[i:i+len(n_gram_head)] == n_gram_head]
                    if len(pos) == 0:
                        gen_x_success = True
                        break
                    else:
                        # remove the n_gram_head from x
                        # get all positions of the n_gram_head in x
                        for p in reversed(pos):
                            # remove len(n_gram_head) elements from x starting from p
                            x = x[:p] + x[p+len(n_gram_head):]
                        # fill the rest of the sequence with random elements
                        x.extend(np.random.choice(vocab, self.n_ctx-self.n_gram*2-len(x), replace=True).tolist())
                x_test = " ".join([str(xi) for xi in x])
                if count(x_test, str_n_gram_head) == 0:
                    gen_x_success = True

            x_test = x + n_gram_head
            # check if there's only one n_gram_head in the sequence
            # to avoid the case where the n_gram_head has 
            # repeated structure such as x= [1, 2, 3, 1] , n_gram_head = [1, 1]
            str_x_test = " "+" ".join([str(xi) for xi in x_test])+ " "
            if count(str_x_test, str_n_gram_head) > 1:
                print("Error in gen_x")
                print(f"str_x_test: {str_x_test}", f"str_n_gram_head: {str_n_gram_head}", 
                      "count: ", count(str_x_test, str_n_gram_head))
            if count(str_x_test, str_n_gram_head) == 1:
                return x
            else:
                return None
        def insert_n_gram_head(x):
            pos = random.randint(0, len(x)-1)
            y = x[pos]
            x_new = x[:pos] + n_gram_head + x[pos:] + n_gram_head
            str_x_new = " "+" ".join([str(xi) for xi in x_new])+" "

            if count(str_x_new, str_n_gram_head) == 2:
                return x_new, y
            else:
                return None, None
        vocab_size = len(self.tokenizer)
        vocab = list(range(vocab_size))
        # set a deterministic n_gram_head
        n_gram_head = list(range(self.n_gram))
       
        str_n_gram_head = " "+" ".join([str(xi) for xi in n_gram_head])+" "
        assert self.n_gram*2 < self.n_ctx, "n_gram*2 should be less than n_ctx"
        success = False
        while not success:
            x = gen_x()
            if x is not None:
                for _ in range(10):
                    x_new, y = insert_n_gram_head(x)
                    if x_new is not None:
                        success = True
                        break
        return x_new, y
            
    def data_gen(self):
        x = []
        y = []
        # get previous random status and recover after generating the dataset
        random_status = random.getstate()
        random.seed(self.seed)
        for i in range(self.num_examples):
            if i % 1000 == 0:
                print(f"Generating example {i}")
            xi, yi = self.gen_single_example()
            x.append(xi)
            y.append(yi)
        x = np.array(x)
        y = np.array(y)
        random.setstate(random_status)
        return x, y
    def split(self, x, y, train_ratio = 0.8):
        num_train = int(len(x)*train_ratio)
        train_x = x[:num_train]
        train_y = y[:num_train]
        test_x = x[num_train:]
        test_y = y[num_train:]
        return train_x, train_y, test_x, test_y


class Random_tokenizer:
    def __init__(self, vocab=None, vocab_size = None) -> None:
        """ The init function of the tokenizer class.
         one of vocab or vocab_size should be provided.
         If vocab is provided, vocab_size will be ignored.
         If vocab is not provided, vocab_size should be provided. we will generate a random vocab of vocab_size."""
        if vocab is not None:
            self.vocab = vocab
            self.vocab_size = len(vocab)
        elif vocab_size is not None:
            self.vocab_size = vocab_size
            self.vocab = [str(i) for i in range(vocab_size)]
        else:
            raise ValueError("one of vocab or vocab_size should be provided.")
        self.vocab_dict = {v: i for i, v in enumerate(self.vocab)}
        self.vocab_dict_inv = {i: v for i, v in enumerate(self.vocab)}
    def encode(self, x):
        """ Encode a string into a list of integers """
        return [self.vocab_dict[i] for i in x]
    def decode(self, x):
        """ Decode a list of integers into a string """
        return ' '.join([self.vocab_dict_inv[i] for i in x])
    def __len__(self):
        return self.vocab_size
    def __getitem__(self, i):
        return self.vocab[i]
    def __iter__(self):
        return iter(self.vocab)
    def __contains__(self, x):
        return x in self.vocab
    def __repr__(self):
        return f"Random_tokenizer(vocab_size={self.vocab_size})"
    def __str__(self):
        return f"Random_tokenizer(vocab_size={self.vocab_size})"
    def __call__(self, x):
        return self.encode(x)
        
# self attention block
class Block(nn.Module):
    def __init__(self, embed_dim, max_len=11):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.c_attn = nn.Linear(embed_dim, embed_dim*3)
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))
    def forward(self, x):
        T = x.size(1)
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
# TODO: RoPEBlock
class RoPEBlock(nn.Module):
    def __init__(self, embed_dim, max_len=11):
        super(RoPEBlock, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.c_attn = nn.Linear(embed_dim, embed_dim*3)
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))
        self.rotary_emb = RotaryEmbedding(dim = embed_dim)
    def forward(self, x):
        T = x.size(1)
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y
    
    
class BaseNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, 
                 is_pe = False, max_len=11, 
                 attn_layers=2, block=None,
                 **kwargs):
        super(BaseNet, self).__init__()
        if block is None:
            raise ValueError("block type should be provided.")
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.is_pe = is_pe
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pe = nn.Embedding(max_len, embed_dim) if is_pe else None
        self.att = nn.ModuleList([block(embed_dim, max_len, **kwargs) for _ in range(attn_layers)])
        self.ln = nn.ModuleList([LayerNorm(embed_dim, True) for _ in range(attn_layers)])
        self.head = nn.Linear(embed_dim, vocab_size)
    
        print(f"BaseNet with {attn_layers} layers of {block} blocks")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Positional Encoding: {is_pe}")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Context length: {max_len}")
        
    def forward(self, x):
        b, t = x.size()
        x = self.embed(x)
        if self.is_pe:
            pos = torch.arange(0, t, dtype=torch.long, device=x.device)
            pe_emb = self.pe(pos) if self.is_pe else 0
            x = x + pe_emb
        for layer, ln in zip(self.att, self.ln):
            x = ln(layer(x))
        x = self.head(x)
        return x
    
# TODO: Rotary PE (RoPE)
class RoPENet(nn.Module):
    def __init__(self, vocab_size, embed_dim, 
                 max_len=11, 
                 attn_layers=2,
                 **kwargs):
        super(RoPENet, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.att = nn.ModuleList([RoPEBlock(embed_dim, max_len, **kwargs) for _ in range(attn_layers)])
        self.ln = nn.ModuleList([LayerNorm(embed_dim, True) for _ in range(attn_layers)])
        self.head = nn.Linear(embed_dim, vocab_size)
    
        print(f"BaseNet with {attn_layers} layers of {RoPEBlock} blocks")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Positional Encoding: RoPE")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Context length: {max_len}")
        
    def forward(self, x):
        b, t = x.size()
        x = self.embed(x)
        
        for layer, ln in zip(self.att, self.ln):
            x = ln(layer(x))
        x = self.head(x)
        return x
    
# TODO: Mamba 2-layer
class MambaNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, 
                 max_len=11, 
                 attn_layers=2,
                 **kwargs):
        super(MambaNet, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        config = MambaConfig(d_model=embed_dim, n_layers=attn_layers)
        self.att = nn.ModuleList([MambaBlock(config) for _ in range(attn_layers)])
        self.ln = nn.ModuleList([LayerNorm(embed_dim, True) for _ in range(attn_layers)])
        self.head = nn.Linear(embed_dim, vocab_size)
    
        print(f"MambaNet with {attn_layers} layers of Mamba blocks")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Positional Encoding: None")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Context length: {max_len}")
        
    def forward(self, x):
        x = self.embed(x)
        
        for layer, ln in zip(self.att, self.ln):
            x = ln(layer(x))
        x = self.head(x)
        return x
    

# TODO: Hybrid-A 2-layer
class HybridA(nn.Module):
    def __init__(self, vocab_size, embed_dim, 
                 max_len=11, 
                 attn_layers=2,
                 **kwargs):
        super(HybridA, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        config = MambaConfig(d_model=embed_dim, n_layers=attn_layers)
        self.att = nn.ModuleList([MambaBlock(config), Block(embed_dim, max_len)])
        self.ln = nn.ModuleList([LayerNorm(embed_dim, True) for _ in range(attn_layers)])
        self.head = nn.Linear(embed_dim, vocab_size)
    
        print(f"HybridA with {attn_layers} layers of blocks")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Positional Encoding: None")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Context length: {max_len}")
        
    def forward(self, x):
        x = self.embed(x)
        
        for layer, ln in zip(self.att, self.ln):
            x = ln(layer(x))
        x = self.head(x)
        return x
    

# TODO: Hybrid-B 2-layer
class HybridB(nn.Module):
    def __init__(self, vocab_size, embed_dim, 
                 max_len=11, 
                 attn_layers=2,
                 **kwargs):
        super(HybridB, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        config = MambaConfig(d_model=embed_dim, n_layers=attn_layers)
        self.att = nn.ModuleList([MambaBlock(config), Block(embed_dim, max_len), MambaBlock(config)])
        self.ln = nn.ModuleList([LayerNorm(embed_dim, True) for _ in range(attn_layers)])
        self.head = nn.Linear(embed_dim, vocab_size)
    
        print(f"HybridA with {attn_layers} layers of blocks")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Positional Encoding: None")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Context length: {max_len}")
        
    def forward(self, x):
        x = self.embed(x)
        
        for layer, ln in zip(self.att, self.ln):
            x = ln(layer(x))
        x = self.head(x)
        return x