import os
import sys
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from EBT.data.nlp.programming_dataloader import ProgrammingDataset
from EBT.data.nlp.collator import NLP_HF_Collator
from EBT.model.nlp.baseline_transformer import Baseline_Transformer_NLP
from EBT.model.nlp.ebt import EBT_NLP

print(ProgrammingDataset)

