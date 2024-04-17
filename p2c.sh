#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python run_p2c.py
CUDA_VISIBLE_DEVICES=3 python test_p2c.py
