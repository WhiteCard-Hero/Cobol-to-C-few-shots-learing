from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__" :
    # path = './' + input('filename: ')
    # file = open(path)
    model = AutoModelForCausalLM.from_pretrained('bigcode/starcoder2-3b', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoder2-3b')

    tokenized_input = tokenizer( 'Hello world in Java', return_tensors='pt' ).to('cuda')
    generated_ids = model.generate( **tokenized_input )
    print(tokenizer.batch_decode(generated_ids)[0])
