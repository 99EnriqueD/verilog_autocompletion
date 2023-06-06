# Based on: https://huggingface.co/course/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch

from dataclasses import dataclass, field
from datetime import datetime
import logging
import torch
import os

from transformers.integrations import TensorBoardCallback
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    CodeGenModel, 
    AutoConfig, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)


@dataclass
class DataArguments:
    train_csv: str = field(
        metadata={'help': 'CSV file containing training data'}
    )
    val_csv: str = field(
        metadata={'help': 'CSV file containing validation data'}
    )
    code_column_name: str = field(
        default='code',
        metadata={'help': 'CSV column name that contains code data for training and validation'}
    )

@dataclass
class CustomTrainArguments:
    model_path: str = field(
        default=None,
        metadata={'help': "Directory where model will be loaded from, default is training a model from scratch (no pretrained-weights with default config https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/codegen#transformers.CodeGenConfig)"}
    )
    context_length: int = field(
        default=512,
        metadata={'help': "Number of tokens used in a training chunk (see https://huggingface.co/course/chapter7/6?fw=pt#preparing-the-dataset)."}
    )
    early_stop_patience: int = field(
        default=3,
        metadata={'help': "Maximum number of epochs to train after lowest val loss measured."}
    )

def main():
    # Params
    parser = HfArgumentParser((DataArguments, CustomTrainArguments, TrainingArguments))
    data_args, training_args, default_training_args = parser.parse_args_into_dataclasses()
    print(f"Data args: {data_args}")
    print(f"Training args: {training_args}")

    default_model_path = "Salesforce/codegen-350M-multi"
    torch.cuda.empty_cache()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO, # Apparently INFO shows info, debug, warnings, and errors? (ERROR only prints warnings and errors it seems?)
        datefmt="[%X]",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(default_training_args.output_dir + "/logs.log",mode="a+"),
            logging.StreamHandler()
        ],
    )

    # Load dataset
    dataset = load_dataset('csv', data_files={'train': data_args.train_csv, 'val': data_args.val_csv})
    # Pre-processing
    tokenizer = AutoTokenizer.from_pretrained(default_model_path) # tokenizer used by all codegen models is the same (also use this one for training from scratch)

    def tokenize(element):
        outputs = tokenizer(
            element[data_args.code_column_name],
            truncation=True,
            max_length=training_args.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == training_args.context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=dataset['train'].column_names)

    # Model
    if training_args.model_path:
        model = AutoModelForCausalLM.from_pretrained(training_args.model_path)
    else:
        logging.info("Initializing model from scratch!")
        model = CodeGenModel(AutoConfig.from_pretrained(default_model_path)) # Uses config defaults 

    # Data collation
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training arguments and setup
    start = datetime.now()

    # Training
    trainer = Trainer(
        model=model,
        args=default_training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.remove_callback(TensorBoardCallback)
    trainer.train()

    later = datetime.now()
    logging.info(" ".join(["Done training at", later.strftime("%d/%m/%Y %H:%M:%S"), "total_runtime:",str(later-start),"saving model..."]))
    model.save_pretrained(os.path.join(default_training_args.output_dir, "last_model"))
    logging.info(" ".join(["Last model saved at:",os.path.join(default_training_args.output_dir, "last_model")]))

if __name__ == "__main__":
    main()