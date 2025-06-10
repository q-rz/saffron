import accelerate
import gc
import os, os.path as osp
import pickle
from argparse import ArgumentParser
from tqdm import tqdm, trange
from typing import *
from typing_extensions import Self
from collections import defaultdict
from copy import copy, deepcopy
import json
import time
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from deepspeed.profiling.flops_profiler import FlopsProfiler