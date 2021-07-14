#!/usr/bin/env bash

python3.8 evaluate_trained_policies_pd.py --hf_smoothness 1
python3.8 evaluate_trained_policies_pd.py --hf_smoothness 0.9
python3.8 evaluate_trained_policies_pd.py --hf_smoothness 0.8
python3.8 evaluate_trained_policies_pd.py --hf_smoothness 0.7
python3.8 evaluate_trained_policies_pd.py --hf_smoothness 0.6
