from dataclasses import dataclass, field
import logging
import torch
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    AutoTokenizer, 
    AutoModelForCausalLM, 
)

@dataclass
class DataArguments:
    test_csv: str = field(
        metadata={'help': 'CSV file containing testing data'}
    )
    unlabeled_column_name: str = field(
        default='code',
        metadata={'help': 'CSV column name that contains code data for testing'}
    )

@dataclass
class EvaluationArguments:
    model_path:str = field(
        metadata={'help': "Directory where model will be loaded from."}
    )
    output_dir:str = field(
        metadata={'help': "Directory where evaluation outputs will be saved"}
    )


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
            logging.FileHandler(eval_args.output_dir + "/perplexity_logs.log",mode="a+"),
            logging.StreamHandler()
        ],
    )

    logging.info(f"Data args: {data_args}")
    logging.info(f"Evaluation args: {eval_args}")
    
    default_model_path = "Salesforce/codegen-350M-multi"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device {device}")

    # Setup dataset
    logging.info("Loading dataset")
    dataset = load_dataset('csv', data_files={'test': data_args.test_csv})

    # Model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(default_model_path, padding_side='left') # tokenizer used by all codegen models is the same (also use this one for training from scratch)
    model = AutoModelForCausalLM.from_pretrained(eval_args.model_path,
        torch_dtype=torch.float16
    )
    model.tie_weights()
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Following: https://huggingface.co/docs/transformers/perplexity
    encodings = tokenizer("\n\n".join(dataset['test'][data_args.unlabeled_column_name]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    logging.info(f"Perplexity: {ppl}")


if __name__ == "__main__":
    main()