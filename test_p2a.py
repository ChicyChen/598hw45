import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from mamba import MambaBlock, Mamba, MambaConfig
from rotary_embedding_torch import RotaryEmbedding
from model import *
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
seed = 0

root = "ckpt_p2_1"

attn_layers = 2 # number of attention layers
embed_dim = 8 # embedding dimension
batch_size = 64 # batch size
n_ctx = 64
vocab_size = 16 # vocabulary size
tokenizer = Random_tokenizer(vocab_size=vocab_size)
model_names = ["Mamba_M4", "Mamba_M8", "Mamba_M16"]
models = []

mamba_model4 = MambaNet(len(tokenizer), embed_dim, max_len=n_ctx*4, attn_layers=attn_layers).to(device)
PATH = os.path.join(root, model_names[0]+".pt")
checkpoint = torch.load(PATH)
mamba_model4.load_state_dict(checkpoint['model_state_dict'])
mamba_model4.eval()
models.append(mamba_model4)

mamba_model8 = MambaNet(len(tokenizer), embed_dim, max_len=n_ctx*4, attn_layers=attn_layers).to(device)
PATH = os.path.join(root, model_names[1]+".pt")
checkpoint = torch.load(PATH)
mamba_model8.load_state_dict(checkpoint['model_state_dict'])
mamba_model8.eval()
models.append(mamba_model8)

mamba_model16 = MambaNet(len(tokenizer), embed_dim, max_len=n_ctx*4, attn_layers=attn_layers).to(device)
PATH = os.path.join(root, model_names[2]+".pt")
checkpoint = torch.load(PATH)
mamba_model16.load_state_dict(checkpoint['model_state_dict'])
mamba_model16.eval()
models.append(mamba_model16)

# generate test data with length 32
# test the model with the test data
num_examples = 20000

n_ctx_list = [16, 32, 64, 128, 256]
M_list = [4, 8, 16]

model_idx = 0
model_test_results = []
for model in models:
    
    dataloader_list = []
    for n_ctx in n_ctx_list:
        dataset = AR(num_examples, tokenizer, 1, M=M_list[model_idx], n_ctx=n_ctx, seed=seed, train_split=0.01)
        test_loader = DataLoader(dataset.test, batch_size=batch_size, shuffle=True)
        dataloader_list.append(test_loader)

    model_name = model_names[model_idx]
    model.eval()
    test_result = []
    for test_loader in dataloader_list:
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x,y = x.to(device), y.to(device)
                y_pred = model(x)[:,-1]
                y_pred = F.softmax(y_pred, dim=-1)
                y_pred = torch.argmax(y_pred, dim=-1)
                correct += (y_pred == y).sum().item()
                total += y.size(0)
            print(f"Test accuracy: {correct/total}")
            test_result.append(correct/total)
    model_test_results.append(test_result)

    plt.figure()
    plt.plot(n_ctx_list, test_result, label=model_name)
    plt.title(f'Test AR Acc')
    plt.xlabel('Len')
    plt.ylabel('Acc')
    plt.legend()
    plt.xticks(n_ctx_list)
    plt.savefig(os.path.join(root, f'AR_Acc_All_M{M_list[model_idx]}.png'))

    model_idx += 1
