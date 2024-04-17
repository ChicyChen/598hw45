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
n_ctx = 64 # training sequence length
num_examples = 100000 # generate 100000 examples
batch_size = 64 # batch size
vocab_size = 16 # vocabulary size
num_epochs = 250    # number of epochs

attn_layers = 2 # number of attention layers
embed_dim = 8 # embedding dimension

tokenizer = Random_tokenizer(vocab_size=vocab_size)

root = "ckpt_p2_2"

dataset4 = AR(num_examples, tokenizer, 1, M=4, n_ctx=n_ctx, seed=seed, train_split=0.8)
dataset8 = AR(num_examples, tokenizer, 1, M=8, n_ctx=n_ctx, seed=seed, train_split=0.8)
dataset16 = AR(num_examples, tokenizer, 1, M=16, n_ctx=n_ctx, seed=seed, train_split=0.8)

dataset_list = [dataset4, dataset8, dataset16]
    
m_val = 4

plt.figure()
for datas in dataset_list:

    train_loader = DataLoader(datas.train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datas.test, batch_size=batch_size, shuffle=True)

    rope_model = RoPENet(len(tokenizer), embed_dim, max_len=n_ctx*4, attn_layers=attn_layers).to(device)

    model_name = "T-RoPE"

    rope_model.train()
    test_list = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rope_model.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = rope_model(x)[:,-1]
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # test over one batch
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred = rope_model(x)[:,-1]
                y_pred = F.softmax(y_pred, dim=-1)
                y_pred = torch.argmax(y_pred, dim=-1)
                correct = (y_pred == y).sum().item()
                test_list.append(correct)
                break
        print(f"epoch {epoch} loss: {total_loss/len(train_loader)}")

    EPOCH = num_epochs
    PATH = os.path.join(root, model_name+"_M"+str(m_val)+".pt")
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': rope_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, PATH)

    
    plt.plot(range(len(test_list)), test_list, label = f"M = {m_val}")
    m_val = m_val*2

    # model.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         x, y = x.to(device), y.to(device)
    #         y_pred = model(x)[:,-1]
    #         y_pred = F.softmax(y_pred, dim=-1)
    #         y_pred = torch.argmax(y_pred, dim=-1)
    #         correct += (y_pred == y).sum().item()
    #         total += y.size(0)
    # print(f"Test accuracy: {correct/total}")

plt.title(f'Test AR Acc')
plt.xlabel('Itr')
plt.ylabel('Acc')
plt.legend()
plt.savefig(os.path.join(root, f'AR_Acc_Itr.png'))

