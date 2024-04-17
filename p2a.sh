#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run_p2a.py
CUDA_VISIBLE_DEVICES=1 python test_p2a.py
