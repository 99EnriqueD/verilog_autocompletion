from dataclasses import dataclass, field
import logging
import torch
import os
import json
import numpy as np
import re

import evaluate
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

@dataclass
class DataArguments:
    test_csv: str = field(
        metadata={'help': 'CSV file containing testing data'}
    )
    input_column_name: str = field(
        default='random_snippet_def',
        metadata={'help':'CSV column name for definition (input) part of supervised pair'}
    )
    label_column_name: str = field(
        default='random_snippet_body',
        metadata={'help':'CSV column name for body (label) part of supervised pair'}
    )

@dataclass
class EvaluationArguments:
    model_path: str = field(
        metadata={'help': "Directory where model will be loaded from, default is training a model from scratch (no pretrained-weights with default config https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/codegen#transformers.CodeGenConfig)"}
    )
    output_dir:str = field(
        metadata={'help': "Directory where evaluation outputs will be saved"}
    )
    beam_size: int = field(
        default=1,
        metadata={'help':'Beam size'}
    )
    skip_special_tokens: bool = field(
        default = True,
        metadata={"help": "Whether or not special tokens should be skipped during decoding."}
    )
    max_length: int = field(
        default=512,
        metadata={'help': 'The maximum number of tokens to generate and maximum number of tokens in input sequence.'}
    )
    batch_size: int = field(
        default=16,
        metadata={'help': "Batch size for prediction/metric computation."}
    )
    metric_subtokenize: bool = field(
        default=True,
        metadata={'help': "Whether or not to subtokenize for metric tokenization."}
    )
    no_repeat_ngram_size:int = field(
        default=4,
        metadata={"help": "Maximum number of times a token that can be generated sequentially."}
    )
    repetition_penalty:float = field(
        default=1.2,
        metadata={"help": "Repetition penalty for generation."}
    )

def condense_spaces(string):
    return re.sub("\s+", " ", string).strip()

def condense_spaces(string):
    return re.sub("\s+", " ", string).strip()

def custom_split(token):
    # from sourcerercc tokenizer.py
  """ Examples:
  custom_split("getAllFilesRecur") --> ['get', 'All', 'Files', 'Recur']
  custom_split("OpenFile") --> ['Open', 'File']
  custom_split("UCI_isLocked") --> ['UCI', 'is', 'Locked']
  custom_split("bLockedOut") --> ['Locked', 'Out']
  custom_split("FileNotFound123") --> ['File', 'Not', 'Found123']
  custom_split("ipv6_IPAddress") --> ['ipv6', 'IP', 'Address']
  custom_split(Address") --> []
  """
  def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    res = [ m.group(0) for m in matches if len(m.group(0)) > 1 ]
    return res if len(res) > 1 else [identifier] # Changed this else to [token] from []!
    
  splits = token.split('_')
  return sum(map(camel_case_split, splits), [])

def encode_with_static_tokenizer(code,subtokens=True):
    separators = [';','(',')','[',']','.',',',":","{","}",'$','"',"'","`", "+", "-", "!", "~", "=", ">", "<", "&", "^", "|", "?", "/"]
    # condense spaces
    code = condense_spaces(code)
    # replace seperators with " "
    for sep in separators:
        code = code.replace(sep, f" {sep} ") # Include the sperators but make sure they don't stick to other things!
    # Split on whitespace
    code = condense_spaces(code)
    tokens = code.split()
    # subtoken tokenization (camelcase)
    
    if subtokens:
        tokens = sum([custom_split(token) for token in tokens], []) # Changed this to keep ordering good for subtokens!
    return tokens


def main():
    # Params
    parser = HfArgumentParser((DataArguments, EvaluationArguments))
    data_args, eval_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO, # Apparently INFO shows info, debug, warnings, and errors? (ERROR only prints warnings and errors it seems?)
        datefmt="[%X]",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(eval_args.output_dir + "/logs.log",mode="a+"),
            logging.StreamHandler()
        ],
    )

    logging.info(f"Data args: {data_args}")
    logging.info(f"Evaluation args: {eval_args}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device {device}")

    default_model_path = "Salesforce/codegen-350M-multi"

    # Setup dataset
    dataset = load_dataset('csv', data_files={'test': data_args.test_csv})

    # Model & tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(default_model_path, padding_side='left') # tokenizer used by all codegen models is the same (also use this one for training from scratch)
    model = AutoModelForCausalLM.from_pretrained(eval_args.model_path,
        torch_dtype=torch.float16
    )
    model.tie_weights()
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    
    def pair_tokenization(examples):
        inputs = [input for input in examples['snippet_def']]
        model_inputs = tokenizer(inputs, max_length=eval_args.max_length, truncation=True,padding='longest')
        labels = tokenizer(examples['snippet_body'],max_length=eval_args.max_length,truncation=True,padding='longest').input_ids
        labels = [torch.tensor(l) for l in labels]
        model_inputs['labels'] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
            pair_tokenization,
            batched=True,
            remove_columns=dataset['test'].column_names,
        )

    # Labeled (chrf, ROUGE, BLEU)
    # From https://discuss.huggingface.co/t/how-to-accessing-the-input-ids-in-evalprediction-predictions-in-seq2seqtrainer/25372
    labeled_metrics = evaluate.combine(
        ["bleu", "chrf", "rouge"]
    )

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        references = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        outputs = labeled_metrics.compute(predictions=predictions,
                                references=references, 
                                 tokenizer=lambda s: encode_with_static_tokenizer(s,eval_args.metric_subtokenize), # Used by both bleu and rouge
                                #  lowercase=True # Used by chrf (optional)
                                )
        return outputs

    trainer_args = Seq2SeqTrainingArguments(
        output_dir="configs",
        predict_with_generate=True,
        per_device_train_batch_size=eval_args.batch_size,
        per_device_eval_batch_size=eval_args.batch_size,
        eval_accumulation_steps=1,
    )
    trainer = Seq2SeqTrainer(model,args=trainer_args, compute_metrics=compute_metrics)
    predictions,labels,metrics = trainer.predict(
        tokenized_dataset['test'],
        max_length=eval_args.max_length*2,
        num_beams=eval_args.beam_size,
        # Can add more generation kwargs here
        no_repeat_ngram_size=eval_args.no_repeat_ngram_size,
        repetition_penalty=eval_args.repetition_penalty,
        )

    metrics_save_file = os.path.join(eval_args.output_dir,"metrics.json")
    json.dump(metrics,open(metrics_save_file,'w+'))
    logging.info(f"Metrics results saved at {metrics_save_file}")

    prediction_save_file = os.path.join(eval_args.output_dir,"predictions.npy")
    np.save(prediction_save_file,predictions)
    logging.info(f"Predictions saved at {prediction_save_file}")

    labels_save_file = os.path.join(eval_args.output_dir,"labels.npy")
    np.save(labels_save_file,labels)
    logging.info(f"Labels saved at {labels_save_file}")


if __name__ == "__main__":
    main()