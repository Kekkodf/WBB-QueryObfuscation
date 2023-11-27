
from transformers import AutoTokenizer, AutoModel
from .abstract_encoder import abstract_encoder
import torch

import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP



class tasb_encoder(abstract_encoder):

    def __init__(self, cache_folder='.', parallel=True):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(cache_folder + 'heka-ai_tasb-bert-50k')
        self.model = AutoModel.from_pretrained(cache_folder + 'heka-ai_tasb-bert-50k')




    def encode(self, input):
        def cls_pooling(model_output, attention_mask):
            return model_output[0][:,0]

        encoded_input = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, cls pooling.
        return np.array(cls_pooling(model_output, encoded_input['attention_mask']))

