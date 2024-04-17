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

root = "ckpt_p3_1"

attn_layers = 2 # number of attention layers
embed_dim = 64 # embedding dimension
batch_size = 64 # batch size
n_ctx = 64
vocab_size = 16 # vocabulary size
tokenizer = Random_tokenizer(vocab_size=vocab_size)
model_name = "Mamba_M16_t10"
models = []

mamba_model = MambaNet(len(tokenizer), embed_dim, max_len=n_ctx*4, attn_layers=attn_layers).to(device)
PATH = os.path.join(root, model_name+".pt")
checkpoint = torch.load(PATH)
mamba_model.load_state_dict(checkpoint['model_state_dict'])
mamba_model.eval()
models.append(mamba_model)


# generate test data with length 32
# test the model with the test data
num_examples = 20000

n_ctx_list = [32, 64, 128, 256]
dataloader_list = []
for n_ctx in n_ctx_list:
    dataset = AR(num_examples, tokenizer, 10, M=16, n_ctx=n_ctx, seed=seed, train_split=0.01)
    # dataset = AR(num_examples, tokenizer, 10, M=16, n_ctx=n_ctx, seed=seed, train_split=0.8)
    test_loader = DataLoader(dataset.test, batch_size=batch_size, shuffle=True)
    dataloader_list.append(test_loader)

model_test_results = []
for model in models:
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
for test_result in model_test_results:
    plt.plot(n_ctx_list, test_result, label=model_name)
plt.title(f'Test AR Acc')
plt.xlabel('Len')
plt.ylabel('Acc')
plt.legend()
plt.xticks(n_ctx_list)
plt.savefig(os.path.join(root, f'AR_Acc_All_2.png'))
