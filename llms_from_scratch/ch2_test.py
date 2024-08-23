import os
import sys
import urllib.request
import re
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

from src import SimpleTokenizerV1
from src import SimpleTokenizerV2
from src import module

def fetch_txt_data(text_file_saving_path):
    if os.path.exists(text_file_saving_path):
        print("File exists:", text_file_saving_path)
        return
    url = ("https://raw.githubusercontent.com/rasbt/""LLMs-from-scratch/main/ch02/01_main-chapter-code/""the-verdict.txt")
    urllib.request.urlretrieve(url, text_file_saving_path)
    return

def load_txt_data(text_file_saving_path):
    if not os.path.exists(text_file_saving_path):
        print("File does not exist:", text_file_saving_path)
        return None
    with open(text_file_saving_path, 'r') as f:
        raw_text = f.read()
    print("total number of characters:", len(raw_text))
    print('--> laod text file done, part context:',raw_text[:99])

    return raw_text

def split_text_to_words(raw_text):
    result = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    result = [item for item in result if item.strip()]
    print(result)
    return result

def test_torch():
    print(torch.__version__)
    print(torch.device('cpu'))
    print('mps available:',torch.backends.mps.is_available())
    return

def token_ids(words):
    all_words = sorted(set(words))
    vocab_size = len(all_words)
    print('vocab size:', vocab_size)
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return vocab

def test_generate_vocab():
    text_file_saving_path = './data/the-verdict.txt'
    fetch_txt_data(text_file_saving_path)
    raw_text = load_txt_data(text_file_saving_path)
    words = split_text_to_words(raw_text)
    vocab = token_ids(words)
    return

if __name__ == '__main__':


    # test for SimpleTokenizerV1
    test_text = 'The brown dog playfully chased the swift fox'
    tokenizer_v1 = SimpleTokenizerV1(vocab)
    print(tokenizer_v1.encode(test_text))



