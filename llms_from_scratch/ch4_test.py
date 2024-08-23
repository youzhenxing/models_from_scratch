import torch
import torch.nn as nn
import tiktoken

from src import DummyGPTModel, LayerNorm
from src import ExampleDeepNeuralNetwork
from src import TransformerBlock
from src import GPTModel

GPT_CONFIG_124M = {"vocab_size": 50257, # Vocabulary size
                   "context_length": 1024, # Context length
                   "emb_dim": 768, # Embedding dimension
                   "n_heads": 12, # Number of attention heads
                   "n_layers": 12, # Number of layers
                   "drop_rate": 0.1, # Dropout rate
                   "qkv_bias": False # Query-Key-Value bias
                   }


def test_dummy_gpt():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every dat holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print('batch:',batch)

    model = DummyGPTModel.DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print('logits:',logits)
    print(logits.shape)


    return


def test_normal_layer():
    torch.manual_seed(123)
    batch_example = torch.randn(2,5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print('out',out)

    mean = out.mean(dim = -1, keepdim = True)
    var = out.var(dim = -1, keepdim = True)
    print('mean',mean)
    print('var',var)
    ln = LayerNorm.LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim = -1, keepdim = True)
    var = out_ln.var(dim = -1, keepdim = True)
    print('mean',mean)
    print('var',var)

    return

def print_gradients(model, x):
    # forward pass
    outputs = model(x)
    target = torch.tensor([[0.]])

    # calculate loss based on how close the target and output are
    loss = nn.MSELoss()
    loss = loss(outputs, target)

    # Backward pass the calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'{name} has gradient mean of {param.grad.abs().mean().item()}')
    return

def test_example_deep_network():
    layer_sizes = [3,3,3,3,3,1]
    sample_input = torch.tensor([1., 0., -1.])
    torch.manual_seed(123)
    model_wo_shortcut = ExampleDeepNeuralNetwork.ExampleDeepNeuralNetwork(layer_sizes, False)
    print_gradients(model_wo_shortcut, sample_input)
    model_w_shortcut = ExampleDeepNeuralNetwork.ExampleDeepNeuralNetwork(layer_sizes, True)
    print_gradients(model_w_shortcut, sample_input)

    return

def test_transformer_block():
    torch.manual_seed(123)
    x = torch.randn(2,4,768)
    block = TransformerBlock.TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print('intput:',x.shape)
    print('output',output)
    return

def test_gpt_model():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every dat holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    model = GPTModel.GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    print('input:',batch)
    print('output:',out)
    print('out shape:',out.shape)

    total_parms = sum(p.numel() for p in model.parameters())
    print('total params:',total_parms)

    total_size_bytes = total_parms * 4
    total_size_mb = total_size_bytes / 1024 / 1024
    print('total_size_mb:',total_size_mb)

    return

if __name__ == '__main__':
    # test_dummy_gpt()
    # test_normal_layer()
    # test_example_deep_network()
    # test_transformer_block()
    test_gpt_model()


