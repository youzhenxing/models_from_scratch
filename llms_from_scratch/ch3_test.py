import os
import sys
import torch
import tiktoken

from torch.utils.data import DataLoader, Dataset
from src import GPTDatasetV1, SelfAttentionV2
from src import module
from src import SelfAttentionV1
from src import SimpleTokenizerV2
from src import CausalAttention

def create_dataloader_v1(text,batch_size=4, max_len=256,
                         stride=128,shuffle=True,drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1.GPTDatasetV1(text,tokenizer,max_len=max_len,stride=stride)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    return dataloader

def test_generate_dataset(text_filename):
    text = ''
    with open(text_filename,'r') as f:
        text = f.read()
    dataloader = create_dataloader_v1(text,batch_size=8,max_len=4,stride=4,shuffle=True,drop_last=True,num_workers=0)
    data_iter = iter(dataloader)
    input_batch,target_batch = next(data_iter)
    print(input_batch,target_batch)
    return dataloader

def test_single_query_scores():
    VOCAB_SIZE = 50227
    OUT_DIMS = 3
    MAX_LEN = 8

    # create dataloader
    raw_text = 'Your journey starts with one step'
    # dataloader = create_dataloader_v1(raw_text,batch_size=8,max_len=MAX_LEN,stride=MAX_LEN,
    #                                   shuffle=True,drop_last=False,num_workers=0)
    # data_iter = iter(dataloader)
    #
    tokenizer = tiktoken.get_encoding('gpt2')
    encode_token = tokenizer.encode(raw_text)


    token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE,OUT_DIMS)
    positions_embedding_layer = torch.nn.Embedding(MAX_LEN,OUT_DIMS)

    #test embedding out
    # first_input,first_target = next(data_iter)
    # print('fist input:',first_input)
    token_embedding = token_embedding_layer(torch.tensor(encode_token))
    position_embedding = positions_embedding_layer(torch.arange(0,len(encode_token)))

    input_embedding = token_embedding + position_embedding
    print(input_embedding)
    print(input_embedding.shape)

    #compute scores for x1
    query = input_embedding[1]
    attn_scores_2 = torch.empty(input_embedding.shape[0])
    for i, x_i in enumerate(input_embedding):
        attn_scores_2[i] = torch.dot(query,x_i)
    print('attn_scores_2:',attn_scores_2,attn_scores_2.shape)
    normalized_attn_scores_2 = attn_scores_2/attn_scores_2.sum()
    print('normalized_attn_scores_2:',normalized_attn_scores_2,normalized_attn_scores_2.sum())
    softmax_normalized_attn_scores2 = torch.softmax(normalized_attn_scores_2,dim=0)
    print('softmax_normalized_attn_scores2:',softmax_normalized_attn_scores2,softmax_normalized_attn_scores2.sum())

    context_z_2 = torch.zeros(query.shape[0])
    for i, x_i in enumerate(input_embedding):
        context_z_2 += softmax_normalized_attn_scores2[i] * x_i
    print('context_z_2:',context_z_2,context_z_2.shape)

    return

def test_query_scores():
    VOCAB_SIZE = 50227
    OUT_DIMS = 3
    MAX_LEN = 8

    # create dataloader
    raw_text = 'Your journey starts with one step'
    # dataloader = create_dataloader_v1(raw_text,batch_size=8,max_len=MAX_LEN,stride=MAX_LEN,
    #                                   shuffle=True,drop_last=False,num_workers=0)
    # data_iter = iter(dataloader)
    #
    tokenizer = tiktoken.get_encoding('gpt2')
    encode_token = tokenizer.encode(raw_text)


    token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE,OUT_DIMS)
    positions_embedding_layer = torch.nn.Embedding(MAX_LEN,OUT_DIMS)

    #test embedding out
    # first_input,first_target = next(data_iter)
    # print('fist input:',first_input)
    token_embedding = token_embedding_layer(torch.tensor(encode_token))
    position_embedding = positions_embedding_layer(torch.arange(0,len(encode_token)))

    input_embedding = token_embedding + position_embedding
    print(input_embedding)
    print(input_embedding.shape)

    #compute scores
    atten_socres = input_embedding @ input_embedding.T

    print('atten_score:',atten_socres,atten_socres.shape)

    normalized_attn_scors  = torch.softmax(atten_socres,dim=-1)
    print('normalized_attn_scors:',normalized_attn_scors,normalized_attn_scors.sum(dim=-1))

    context_z = normalized_attn_scors @ input_embedding
    print('context_z:',context_z,context_z.shape)

    torch.manual_seed(123)
    sa_v1 = SelfAttentionV1.SelfAttention_v1(3,2)
    context_z_1 = sa_v1(input_embedding)
    print('context_z_1:',context_z_1,context_z_1.shape)

    # causal masking
    sa_v2 = SelfAttentionV2.SelfAttention_v2(3,2,False)
    queries = sa_v2.W_query(input_embedding)
    keys = sa_v2.W_keys(input_embedding)
    attns_scores = queries @ keys.T
    attns_weights = torch.softmax(attns_scores / keys.shape[-1] ** 0.5, dim = -1)
    print('attns_weights:',attns_weights,attns_weights.shape)

    context_length = attns_scores.shape[-1]

    mask_simple = torch.tril(torch.ones(context_length,context_length))
    masked_simple = attns_weights * mask_simple
    print('mask_simple:',mask_simple,mask_simple.shape)
    print('masked_simple:',masked_simple,masked_simple.shape)

    row_sum = masked_simple.sum(dim=-1,keepdim=True)
    normalized_masked_simple = masked_simple / row_sum
    print('row_sum:',row_sum,row_sum.shape)
    print('normalized_masked_simple:',normalized_masked_simple,normalized_masked_simple.shape)

    mask = torch.triu(torch.ones(context_length,context_length),diagonal=1)
    attns_scores = attns_scores.masked_fill(mask.bool(),-torch.inf)
    attns_weights = torch.softmax(attns_scores/keys.shape[-1] ** 0.5, dim = -1)
    print(attns_weights,attns_weights.sum(dim=-1),attns_weights.shape)

    # dropout
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6,6)
    print(dropout(example))
    print(dropout(attns_weights))

    # test batch
    batch = torch.stack((input_embedding, input_embedding), dim = 0)
    print(batch.shape,input_embedding.shape)
    print(batch)

    context_length = batch.shape[1]
    ca = CausalAttention.CausalAttention(3,2,context_length, 0.0)
    context_vec = ca(batch)
    print('context_vec:',context_vec,context_vec.shape)

    # multi-heads attn
    context_length = batch.shape[1]
    print('context_length:',context_length)
    d_in, d_out = 3,2
    mha = CausalAttention.MultiHeadAttentionWrapper(3,2,context_length,0.0,2)
    context_vec = mha(batch)
    print('context_vec:',context_vec,context_vec.shape)

    return

def test_self_attn():



    return
def test_embedding(text_filename):
    VOCAB_SIZE = 50227
    OUT_DIMS = 256
    MAX_LEN = 4

    # create dataloader
    with open(text_filename,'r') as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(raw_text,batch_size=8,max_len=MAX_LEN,stride=MAX_LEN,
                                      shuffle=True,drop_last=True,num_workers=0)
    data_iter = iter(dataloader)


    token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE,OUT_DIMS)
    positions_embedding_layer = torch.nn.Embedding(MAX_LEN,OUT_DIMS)

    #test embedding out
    first_input,first_target = next(data_iter)
    print('fist input:',first_input)
    token_embedding = token_embedding_layer(first_input)
    position_embedding = positions_embedding_layer(torch.arange(0,MAX_LEN))

    input_embedding = token_embedding + position_embedding
    print(input_embedding)
    print(input_embedding.shape)

if __name__ ==  "__main__":
   text_file_saving_path = './data/the-verdict.txt'


   module.print_line('test_generate_dataset', True)
   test_generate_dataset(text_file_saving_path)
   module.print_line('test_generate_dataset', False)

   module.print_line('test_embedding', True)
   test_embedding(text_file_saving_path)
   module.print_line('test_embedding', False)

   module.print_line('test_single_query_scores', True)
   test_single_query_scores()
   module.print_line('test_single_query_scores', False)

   module.print_line('test_query_scores', True)
   test_query_scores()
   module.print_line('test_query_scores', False)

   print("--> finish")