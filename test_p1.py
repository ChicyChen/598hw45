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

roots = ["ckpt_p1_1", "ckpt_p1_2"]
for root in roots:

    attn_layers = 2 # number of attention layers
    embed_dim = 8 # embedding dimension
    batch_size = 64 # batch size
    n_ctx = 64
    vocab_size = 16 # vocabulary size
    tokenizer = Random_tokenizer(vocab_size=vocab_size)
    model_names = ["T-PE", "T-RoPE", "T-NoPE", "Mamba", "Hybrid-A", "Hybrid-B"]
    models = []

    attn_model = BaseNet(len(tokenizer), embed_dim, True, max_len=n_ctx*4, attn_layers=attn_layers, block=Block).to(device)
    PATH = os.path.join(root, model_names[0]+".pt")
    checkpoint = torch.load(PATH)
    attn_model.load_state_dict(checkpoint['model_state_dict'])
    attn_model.eval()
    models.append(attn_model)

    rope_model = RoPENet(len(tokenizer), embed_dim, max_len=n_ctx*4, attn_layers=attn_layers).to(device)
    PATH = os.path.join(root, model_names[1]+".pt")
    checkpoint = torch.load(PATH)
    rope_model.load_state_dict(checkpoint['model_state_dict'])
    rope_model.eval()
    models.append(rope_model)

    nope_model = BaseNet(len(tokenizer), embed_dim, False, max_len=n_ctx*4, attn_layers=attn_layers, block=Block).to(device)
    PATH = os.path.join(root, model_names[2]+".pt")
    checkpoint = torch.load(PATH)
    nope_model.load_state_dict(checkpoint['model_state_dict'])
    nope_model.eval()
    models.append(nope_model)

    mamba_model = MambaNet(len(tokenizer), embed_dim, max_len=n_ctx*4, attn_layers=attn_layers).to(device)
    PATH = os.path.join(root, model_names[3]+".pt")
    checkpoint = torch.load(PATH)
    mamba_model.load_state_dict(checkpoint['model_state_dict'])
    mamba_model.eval()
    models.append(mamba_model)

    hyba_model = HybridA(len(tokenizer), embed_dim, max_len=n_ctx*4, attn_layers=attn_layers).to(device)
    PATH = os.path.join(root, model_names[4]+".pt")
    checkpoint = torch.load(PATH)
    hyba_model.load_state_dict(checkpoint['model_state_dict'])
    hyba_model.eval()
    models.append(hyba_model)

    hybb_model = HybridB(len(tokenizer), embed_dim, max_len=n_ctx*4, attn_layers=attn_layers).to(device)
    PATH = os.path.join(root, model_names[5]+".pt")
    checkpoint = torch.load(PATH)
    hybb_model.load_state_dict(checkpoint['model_state_dict'])
    hybb_model.eval()
    models.append(hybb_model)

    # generate test data with length 32
    # test the model with the test data
    num_examples = 20000

    n_ctx_list = [16, 32, 64, 128, 256]
    dataloader_list = []
    for n_ctx in n_ctx_list:
        if root == "ckpt_p1_1":
            dataset = InductionAR(num_examples, tokenizer, 1, n_ctx=n_ctx, seed=seed, train_split=0.01)
        else:
            dataset = InductionAR(num_examples, tokenizer, 5, n_ctx=n_ctx, seed=seed, train_split=0.01)
        test_loader = DataLoader(dataset.test, batch_size=batch_size, shuffle=True)
        dataloader_list.append(test_loader)

    model_idx = 0
    model_test_results = []
    for model in models:
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
    model_idx = 0
    for test_result in model_test_results:
        model_name = model_names[model_idx]
        plt.plot(n_ctx_list, test_result, label=model_name)
        model_idx += 1
    plt.title(f'Test AR Acc')
    plt.xlabel('Len')
    plt.ylabel('Acc')
    plt.legend()
    plt.xticks(n_ctx_list)
    plt.savefig(os.path.join(root, f'AR_Acc_All.png'))
