import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score

LABEL_COLUMNS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval","disgust",
    "embarrassment","excitement","fear","gratitude","grief","joy","love",
    "nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
]

def load_and_prep(csv_paths):
    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    df = df.dropna(subset=["text"])
    return df

def compute_metrics(pred):
    logits, labels = pred
    # Move tensors to CPU and convert logits to probabilities using sigmoid for multi-label classification
    if torch.cuda.is_available():
        logits = torch.tensor(logits).cuda()
    else:
        logits = torch.tensor(logits)
    
    probs = torch.sigmoid(logits)
    # Apply threshold to get binary predictions
    preds = (probs > 0.5).int().cpu().numpy()
    
    # Convert labels to numpy if they're tensors and ensure they're integers
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels = labels.astype(int)
    
    # Calculate macro F1 score
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    return {"f1_macro": f1_macro}

def main():
    # Check GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    df = load_and_prep(["data/goemotions_1.csv","data/goemotions_2.csv","data/goemotions_3.csv"])
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_ds = Dataset.from_pandas(train_df)
    test_ds  = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    def tokenize_and_prepare_labels(batch):
        # Tokenize the text
        tokenized = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)
        
        # Prepare labels as float tensors for multi-label classification
        labels = []
        for i in range(len(batch["text"])):
            label_vector = [float(batch[col][i]) for col in LABEL_COLUMNS]
            labels.append(label_vector)
        
        tokenized["labels"] = labels
        return tokenized
    
    train_ds = train_ds.map(tokenize_and_prepare_labels, batched=True)
    test_ds  = test_ds.map(tokenize_and_prepare_labels, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        problem_type="multi_label_classification",
        num_labels=len(LABEL_COLUMNS)
    )
    
    # Move model to GPU if available
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir="./roberta-goemotions",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        per_device_train_batch_size=32,  # Increased batch size for GPU
        per_device_eval_batch_size=64,   # Increased eval batch size
        gradient_accumulation_steps=2,   # Accumulate gradients for effective batch size of 64
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        warmup_steps=500,                # Add warmup for better training stability
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_pin_memory=True,      # Enable pin_memory for GPU
        dataloader_num_workers=4,        # Use multiple workers for data loading
        fp16=True,                       # Enable mixed precision training
        logging_dir="./logs",            # Add logging directory
        report_to="none",                # Disable wandb/tensorboard unless needed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print("Starting training...")
    trainer.train()
    
    print("Starting evaluation...")
    trainer.evaluate()
    
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("./roberta-goemotions")
    
    # Print final GPU memory usage
    if torch.cuda.is_available():
        print(f"GPU memory after training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
