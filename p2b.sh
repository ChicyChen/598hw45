#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python run_p2b.py
CUDA_VISIBLE_DEVICES=2 python test_p2b.py
