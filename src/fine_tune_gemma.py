import torch
import os
import json
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Check CUDA availability
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available. Please make sure CUDA is installed and configured correctly.")

# Check if GPU supports bfloat16
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

# Hugging Face model id
model_id = "google/gemma-3-4b-it"  # Using the image-text version for multimodal training

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager",  # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16,  # What torch dtype to use, defaults to auto
    device_map="auto",  # Let torch decide how to load the model
)

# BitsAndBytesConfig int-4 config
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

# Load model and processor
print("Loading model and processor...")
model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained(model_id)

# Set up LoRA configuration for efficient fine-tuning
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

# Helper function to process vision information in messages
def process_vision_info(messages):
    """
    Process messages to extract image data
    """
    for message in messages:
        if message["role"] == "user":
            for i, content_item in enumerate(message["content"]):
                if isinstance(content_item, dict) and content_item.get("type") == "image":
                    image_path = content_item.get("image")
                    if image_path and os.path.exists(image_path):
                        return image_path  # Return the path to the first image found
    return None

# Load and prepare the dataset
def load_dataset():
    """
    Load the dataset created by create_labeled_data.py
    """
    try:
        with open("oai_formatted_dataset.json", "r") as f:
            data = json.load(f)
        return Dataset.from_list(data)
    except FileNotFoundError:
        print("Dataset file not found. Make sure to run create_labeled_data.py first.")
        exit(1)

# Load the dataset
print("Loading dataset...")
dataset = load_dataset()
print(f"Dataset loaded with {len(dataset)} examples")

# Configure SFT parameters
args = SFTConfig(
    output_dir="gemma-finance-analyst",     # directory to save and repository id
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=5,                        # log every 5 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # use reentrant checkpointing
    dataset_text_field="",                  # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
)
args.remove_unused_columns = False # important for collator

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_path = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        
        # Process image if available
        if image_path:
            from PIL import Image
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Provide a placeholder or default image
                images.append(None)
        else:
            images.append(None)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch

# Create the trainer
print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

if __name__ == "__main__":
    print("Starting training...")
    # Start training
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model()
    print("Training complete!")
