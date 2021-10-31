import os
import sys
import argparse
import yaml
import pandas as pd
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertForTokenClassification, BertTokenizer
import numpy as np
from tqdm import tqdm


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = load_config(args.config_path)

    model, tokenizer = load_ner_model(
        args.model_path, len(config['NER']['tag2idx']))

    notes = load_notes(
        config['NOTEEVENTS'])

    notes = notes[:args.number_documents]

    hadm_ids = notes['HADM_ID'].to_list()[:args.number_documents]

    embeddings = get_embeddings(notes, model,
                                tokenizer, config)

    save_embeddings(config['EMBEDDINGS'], embeddings, hadm_ids)


def load_config(config_path: str) -> dict:
    '''
    Load configuration file
    '''
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
        except:
            raise FileNotFoundError('Could not find configuration')
    return config


def load_ner_model(model_path: str, num_labels: int):
    '''
    Loads trained i2b2 2010 ner model + tokenizer
    '''
    print('loading model')
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
    model = BertForTokenClassification.from_pretrained(
        model_path, num_labels=num_labels)
    print('loaded')
    return model, tokenizer


def load_notes(file_path: str) -> pd.DataFrame:
    '''
    Loads clinical descriptions json which contains
    sentences to label
    '''
    return pd.read_csv(file_path)[['HADM_ID', 'TEXT']]


def get_embeddings(notes: pd.DataFrame, model, tokenizer, config: dict):
    '''
    Get sentence embeddings
    '''
    embeddings = []
    for row in tqdm(notes.itertuples(), total=notes.shape[0], position=0, leave=True):
        sent_embeddings = _get_embedding(
            row[2], model, tokenizer, config['NER']['max_len'])
        embeddings.append(sent_embeddings)
    return np.vstack([embeddings])


def _get_embedding(text: str, model, tokenizer, max_len: int) -> np.ndarray:
    '''
    Make NER predictions
    '''
    tokenized_texts = []
    temp_token = []
    temp_token.append('[CLS]')
    token_list = tokenizer.tokenize(text)

    for _, token in enumerate(token_list):
        temp_token.append(token)

    # Trim the token to fit the length requirement
    if len(temp_token) > max_len-1:
        temp_token = temp_token[:max_len-1]
    temp_token.append('[SEP]')
    tokenized_texts.append(temp_token)
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype="long", truncating="post", padding="post")

    # For fine tune of predict, with token mask is 1,pad token is 0
    attention_masks = [[int(i > 0) for i in ii] for ii in input_ids]
    segment_ids = [[0] * len(input_id) for input_id in input_ids]
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    segment_ids = torch.tensor(segment_ids)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None,
                        attention_mask=None, output_hidden_states=True)
        hidden_states = outputs[1]
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        sentence_embedding = torch.nn.functional.normalize(
            sentence_embedding, p=2, dim=0).detach().cpu().numpy()
    return sentence_embedding


def save_embeddings(embeddings_config: dict, embeddings: np.ndarray, hadm_ids: list):
    '''
    Save sentence embeddings as .npz
    '''
    np.savez_compressed(os.path.join(embeddings_config['output_path'], embeddings_config['filename']), embeddings=embeddings,
                        hadm_ids=hadm_ids)
    print(f'Embeddings saved!')


def parse_args(args):
    epilog = """
    Examples of usage
    python3 embeddings.py --config_path configuration/cfg.yml --model_path ../models/clinical_bert/
    """
    description = """
    Compute sentence embeddings from hidden_states of i2b2 NER system
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog)
    parser.add_argument('--config_path', help='Path to the config', type=str),
    parser.add_argument('--model_path',
                        help='Path to the model for making inferences',
                        type=str)
    parser.add_argument('--number_documents',
                        help='Number of documents to process',
                        type=int)                    
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
