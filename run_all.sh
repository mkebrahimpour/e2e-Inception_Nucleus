#!/usr/bin/env bash
rm *.out
rm *.tsv
export CUDA_VISIBLE_DEVICES=1; python model_run.py 
