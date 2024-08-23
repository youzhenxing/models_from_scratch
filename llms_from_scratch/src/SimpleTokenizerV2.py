import os
import sys
import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:str for str,i in vocab.items()}
    def encode(self,raw_text):
        result = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
        result = [item.strip() for item in result if item.strip()]
        result = [item if item in self.str_to_int.keys() else "<|unk|>" for item in result]
        ids = [self.str_to_int[s] for s in result]
        return ids
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text