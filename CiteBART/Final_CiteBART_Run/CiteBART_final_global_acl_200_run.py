import os
import re
from datasets import DatasetDict, Dataset
from transformers import (BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments,
                          BartConfig, GenerationConfig, DataCollatorForSeq2Seq)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Any
import pandas as pd
import math
import numpy as np
import time
import traceback

# Read dataset
def read_dataset():
    train_df = pd.read_csv(train_dataset_path)
    train_set = []

    for _, i in train_df.iterrows():
        temp_citing_title = i['citing_title']
        temp_citing_abstract = i['citing_abstract']
        temp_masked_context = i['masked_cit_context'].replace("OTHERCIT", "")  # "Fill the mask with an appropriate citation: " +

        temp_train_input = temp_citing_title + " </s> " + temp_citing_abstract + " </s> " + temp_masked_context

        temp_dict = {"masked_cit_context": temp_train_input, 
                     "masked_token_target": i['masked_token_target']}

        train_set.append(temp_dict)

    eval_df = pd.read_csv(final_eval_dataset_path)
    eval_set = []

    for _, i in eval_df.iterrows():
        temp_citing_title = i['citing_title']
        temp_citing_abstract = i['citing_abstract']
        temp_masked_context = i['masked_cit_context'].replace("OTHERCIT", "")  # "Fill the mask with an appropriate citation: " +

        temp_eval_input = temp_citing_title + " </s> " + temp_citing_abstract + " </s> " + temp_masked_context

        temp_dict = {"masked_cit_context": temp_eval_input,
                     "masked_token_target": i['masked_token_target']}

        eval_set.append(temp_dict)

    return train_set, eval_set

# Preprocessing function (Okay)
def preprocess_function(examples):
    inputs = [example.replace("<mask>", "<extra_id_0>", 1).replace("<mask>", "").replace("<extra_id_0>", "<mask>")
              for example in examples["masked_cit_context"]]
    targets = [example for example in examples["masked_token_target"]]

    model_inputs = tokenizer(inputs, max_length=max_token_limit, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_token_limit, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Self-Explanatory
def add_spaces_after_commas(text: str) -> str:
    return ', '.join(part.strip() for part in text.split(','))  #  NEW!!!!

# Fill the mask by generating (BART-thing) Okay
def fill_mask(sentence, num_predictions):

    # Read CSV and extract relevant column as a list (Optimized)
    #all_cit_list = pd.read_csv(all_citations_path)['citation_items'].tolist()

    # Tokenize input
    sentence = sentence.replace("<mask>", "<extra_id_0>", 1).replace("<mask>", "").replace("<extra_id_0>", "<mask>")
    input_ids = tokenizer.encode(sentence, return_tensors="pt", max_length=max_token_limit, truncation=True, padding="max_length").to("cuda")

    # Generate outputs
    outputs = model.generate(input_ids, generation_config=cit_generation_config)

    # Decode outputs efficiently
    predictions = [add_spaces_after_commas(tokenizer.decode(output, skip_special_tokens=True).strip()) for output in outputs]

    # Get unique predictions
    unique_predictions: List[Any] = list(dict.fromkeys(predictions))  # Remove duplicates while preserving order

    # Print the top n predictions
    # for i, pred in enumerate(unique_predictions, num_predictions):
    #     print(f"Prediction {i}: {pred} \n\n")

    # Ensure at least `num_predictions` predictions
    if len(unique_predictions) < num_predictions:
        unique_predictions.extend([unique_predictions[-1]] * (num_predictions - len(unique_predictions)))

    return unique_predictions[:num_predictions]

# Calculate for metrics of one instance
def compare_pred_with_correct_value(predictions, ground_truth, k=10):
    """Compute Recall@k, MRR, and NDCG for a single ground truth citation."""

    recall_at_k = 0
    reciprocal_rank = 0
    ndcg = 0

    #print(f"Predictions: {predictions}")
    #print(f"Number of Predictions: {len(predictions)})
    #print(f"Ground Truth: {ground_truth}")

    # Normalize ground truth
    ground_truth = ground_truth.replace(" and ", " ").replace(" et al.,", "").replace(",", "").strip()
    truth_tokens = ground_truth.split()

    #print(f"Truth Tokens: {truth_tokens}")

    # Ensure at least two tokens for comparison
    if len(truth_tokens) < 2:
        return recall_at_k, reciprocal_rank, ndcg # No valid NDCG if no valid ground truth
    
    # Check if the ground truth appears in the top-k predictions
    for p_idx, prediction in enumerate(predictions[:k]):  # Consider only top-k predictions
        if all(token in prediction for token in truth_tokens):  # Check if any token matches
            recall_at_k = 1  # Since there's only one correct answer
            reciprocal_rank = 1 / (p_idx + 1)
            ndcg = 1 / np.log2(p_idx + 2)  # Compute DCG for rank position
            break  # No need to check further once we find the match
        
    #print(f"Recall@k: {recall_at_k}, RR: {reciprocal_rank}, NDCG: {ndcg}")

    return recall_at_k, reciprocal_rank, ndcg

 
 # Summarize the numbers

def calc_eval_metrics(val_dataset, k=10):
    recall_list = []
    reciprocal_rank_list = []
    ndcg_list = []
    
    start_time = time.time()  # Start time tracking

    print("=== Evaluation Started ===")  # Log start of evaluation

    for idx, e in enumerate(val_dataset):
        masked_cit_context = e["masked_cit_context"]
        target_token = e["masked_token_target"]

        temp_predictions = fill_mask(masked_cit_context, k)  # Get top-k predictions
        
        # Compute Recall@k, MRR, and NDCG@k
        recall_at_k, temp_reciprocal_rank, temp_ndcg = compare_pred_with_correct_value(temp_predictions, target_token, k)
        
        recall_list.append(recall_at_k)
        reciprocal_rank_list.append(temp_reciprocal_rank)
        ndcg_list.append(temp_ndcg)
        
        # Print running averages every 500 iterations
        if (idx + 1) % 500 == 0:
            running_mean_recall = np.mean(recall_list) if recall_list else 0
            running_mean_reciprocal_rank = np.mean(reciprocal_rank_list) if reciprocal_rank_list else 0
            running_mean_ndcg = np.mean(ndcg_list) if ndcg_list else 0
            
            elapsed_time = time.time() - start_time  # Calculate elapsed time from the start

            print(f"After {idx+1} entries - Running Mean Recall@{k}: {running_mean_recall:.4f}, MRR@{k}: {running_mean_reciprocal_rank:.4f}, NDCG@{k}: {running_mean_ndcg:.4f}, Elapsed Time: {elapsed_time:.2f} seconds")
    
    # Compute Mean Metrics
    mean_recall_at_k = np.mean(recall_list) if recall_list else 0
    mean_reciprocal_rank_at_k = np.mean(reciprocal_rank_list) if reciprocal_rank_list else 0
    mean_ndcg_at_k = np.mean(ndcg_list) if ndcg_list else 0

    end_time = time.time()  # End time tracking
    elapsed_time = end_time - start_time
    
    # Log final summary
    print(f"\n=== Final Evaluation Metrics ===\n")
    print(f"Mean Recall@{k}: {mean_recall_at_k:.4f}\n")
    print(f"Mean MRR@{k}: {mean_reciprocal_rank_at_k:.4f}\n")
    print(f"Mean NDCG@{k}: {mean_ndcg_at_k:.4f}\n")
    print(f"Total Evaluation Time: {elapsed_time:.2f} seconds\n")
    print(f"=============================\n")

    # Create a DataFrame for better readability
    metrics_data = {
        "Metric": [f"Recall@{k}", f"Mean Reciprocal Rank@{k}", f"Normalized Discounted Cumulative Gain@{k}"],
        "Value": [mean_recall_at_k, mean_reciprocal_rank_at_k, mean_ndcg_at_k]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    print("\n=======>>> Evaluation Metrics Summary\n")
    print(metrics_df.to_string(index=False))

    return metrics_df  # Optionally return metrics for logging

### Code to Run

dataset_folder = "./dataset"  # Change this to your actual dataset path
train_dataset_path = f"{dataset_folder}/final_cleaned_acl_global_context_dataset_train.csv"
eval_dataset_path = f"{dataset_folder}/final_cleaned_acl_global_context_dataset_eval.csv"

final_eval_dataset_path = "/home/ubuntu/[March 27, 2025] Final Dataset/sampled_final_cleaned_acl_global_context_dataset_eval.csv"

# Just to see if the change is drastic
old_train_dataset_path = "/home/ubuntu/CiteBART/acl_200_global_rerun/dataset/cleaned_acl_global_context_dataset_train.csv"
old_eval_dataset_path = "/home/ubuntu/CiteBART/acl_200_global_rerun/dataset/cleaned_acl_global_context_dataset_eval.csv"

# Arxiv eval to prove that the model only works in one specific dataset
arxiv_eval_dataset_path = "/home/ubuntu/CiteBART/Final_CiteBART_Run/dataset/arxiv_context_dataset_eval.csv"

print("Datasets loaded.")

custom_model_name = "CiteBART_final_global_ACL200_run_1245_March-27_P3"
checkpoints_location = f"./checkpoints/{custom_model_name}"
model_save_location = f"./models/{custom_model_name}"

# Ensure directories exist
os.makedirs(checkpoints_location, exist_ok=True)
os.makedirs(model_save_location, exist_ok=True)

print("Checkpoint and Model Saves directories created successfully.")

pretrained_model_name_or_path = "facebook/bart-base"
max_token_limit = 350

# Initialize the config
config = BartConfig.from_pretrained(pretrained_model_name_or_path, attention_dropout=0.123)

# Initialize the tokenizer
tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path, truncation=True,
                                          padding='max_length', model_max_length=max_token_limit)

# Set up the model
model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, config=config)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

cit_generation_config = GenerationConfig.from_model_config(model.config)

cit_generation_config.max_new_tokens = 25
cit_generation_config.do_sample = False
cit_generation_config.top_k = 50
cit_generation_config.num_return_sequences = 20
cit_generation_config.early_stopping = False
cit_generation_config.num_beams = 20
cit_generation_config.forced_bos_token_id = 0

cit_generation_config.num_beam_groups = 10
cit_generation_config.diversity_penalty = 1.5

print("BART Configurations loaded.")

train_dataset, eval_dataset = read_dataset()

data = {
    "train": train_dataset,
    "eval": eval_dataset
}

# Convert to Dataset
train_dataset = Dataset.from_pandas(pd.DataFrame(data["train"]))
validation_dataset = Dataset.from_pandas(pd.DataFrame(data["eval"]))

dataset = DatasetDict({
    "train": train_dataset,
    "eval": validation_dataset
})

# Preprocess the datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True)

print("Datasets tokenized successfully.")

num_epochs = 15
warmup_steps = 500
train_and_eval_batch_sizes = 16
auto_find_batch_size_flag = True
skip_training = False

training_args = TrainingArguments(
    output_dir=checkpoints_location,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="steps",  # Change to 'steps' to control logging manually
    logging_steps=1000,  # Adjust the logging frequency (e.g., log every 500 steps)
    warmup_steps=warmup_steps,
    save_strategy="epoch",
    save_total_limit=15,
    resume_from_checkpoint= False,
    disable_tqdm=True
)

if auto_find_batch_size_flag:
    training_args.auto_find_batch_size = True
else:
    training_args.per_device_train_batch_size = train_and_eval_batch_sizes
    training_args.per_device_eval_batch_size = train_and_eval_batch_sizes

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("Training parameters initialized.")

print("Starting training...or not?")

# List checkpoint folders containing the word "checkpoint"
checkpoint_dirs = [d for d in os.listdir(checkpoints_location) if "checkpoint" in d]
print(f"Found {len(checkpoint_dirs)} checkpoint folder(s) out of {num_epochs} epochs expected.")

# If training is complete
if len(checkpoint_dirs) == num_epochs:
    print("Skipping training. All epochs are checkpointed. Loading model from checkpoint...")

    # Find the checkpoint folder with the highest numeric suffix
    best_checkpoint = None
    best_value = -1
    for d in checkpoint_dirs:
        # Assuming folder name format like "checkpoint-<number>"
        match = re.search(r'checkpoint-(\d+)', d)
        if match:
            value = int(match.group(1))
            if value > best_value:
                best_value = value
                best_checkpoint = d

    if best_checkpoint is None:
        raise ValueError("No valid checkpoint folder found.")
    
    # Build full path to the best checkpoint
    best_checkpoint_path = os.path.join(checkpoints_location, best_checkpoint)
    print(f"Loading model from: {best_checkpoint_path}")

    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(best_checkpoint_path)
else:
    # Start training with error handling
    try:
        start_time = time.time()  # Track start time
        
        trainer.train(resume_from_checkpoint=False)

        # Save model checkpoint after each epoch
        trainer.save_model(model_save_location)
        tokenizer.save_pretrained(model_save_location)

        # Extract training metrics
        metrics = trainer.state.log_history
        metrics_table = pd.DataFrame(metrics)

        # Log final metrics
        metrics_log = metrics_table.to_string(index=False)
        print("Training Completed Successfully ✅\n\nMetrics:\n" + metrics_log)

        end_time = time.time()  # Track end time
        elapsed_time = end_time - start_time
        print(f"Total Training Time: {elapsed_time:.2f} seconds")

    except Exception as e:
        error_message = f"Training Failed ❌\n\nError Details:\n{traceback.format_exc()}"

        # Re-raise the error so it doesn't silently fail
        raise

print("Beginning Evaluation...")

eval_results = trainer.evaluate()
print(f"\n*****************\nEval loss after fine-tuning: {eval_results['eval_loss']}\n"
      f"Perplexity after fine-tuning: {math.exp(eval_results['eval_loss']):.2f}\n\n")

calc_eval_metrics(eval_dataset)

print("Done evaluating.")
