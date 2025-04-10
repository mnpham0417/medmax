import json
import argparse
from tqdm import tqdm
from tokenizers import Tokenizer

def offset_image_tokens(tokens):
    ## Chameleon image tokens range from 4-8195 instead of 0-8191
    offset = 4
    tokens = [x + offset for x in tokens]
    return tokens

def main(args):

    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    tokenizer.bos = 0
    tokenizer.pad = 1
    tokenizer.eos = 2
    tokenizer.image_id = 8711
    
    input_file = args.input_file
    with open(input_file, 'r') as f:
        input_data = list(f) 

    all_data = []
    for instance in tqdm(input_data):
        instance = eval(instance)
        image_tokens = instance['image_tokens']
        image_tokens = offset_image_tokens(instance['image_tokens'])
        image_tokens = [8197] + image_tokens + [8196] ## add boi and eoi tokens

        # text = instance['text'] ## it should have <image> 
        # text_tokenized = [tokenizer.bos] + tokenizer.encode(text).ids + [tokenizer.eos]
        # new_tokens = []
        # for token in text_tokenized:
        #     if token == tokenizer.image_id:
        #         new_tokens = new_tokens + image_tokens
        #     else:
        #         new_tokens = new_tokens + [token]
        # instance['tokens'] = new_tokens
        instance['tokens'] = image_tokens
        all_data.append(instance)
    
    with open(args.output_file, 'w') as f:
        for instance in tqdm(all_data):
            f.write(json.dumps(instance))
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_file', type = str, help = 'tokenization file')
    parser.add_argument('--input_file', type = str, help = 'input file name')
    parser.add_argument('--output_file', type = str, help = 'output file name')
    args = parser.parse_args()
    main(args)