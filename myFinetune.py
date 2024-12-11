import argparse
import multiprocessing
import os

import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
from trl import SFTTrainer
if __name__ == "__main__" :
    path = "starcoder2"

    os.makedirs(path, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        "bigcode/starcoder2-3b",
        quantization_config=bnb_config,
        device_map="auto",
    )

    data = load_dataset(
        "semeru/code-code-translation-java-csharp",
        split="train",
        num_proc=multiprocessing.cpu_count(),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        max_seq_length=512,
        args=transformers.TrainingArguments(
            bf16="bf16",
            output_dir=path,
            optim="paged_adamw_8bit",
        ),
        peft_config=lora_config,
        dataset_text_field="text",
    )

    trainer.train()
    model.save_pretrained(os.path.join(path, "final_checkpoint/"))
