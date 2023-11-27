import argparse
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2, IOB1
import numpy as np

from nltk.stem.porter import PorterStemmer
import dataset
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Set Argument Parser
parser = argparse.ArgumentParser()
# Training hyperparameters
parser.add_argument('--epoch', type=int, default=20)
# parser.add_argument('--epoch', type=int, default=1) # speed computation for debugging
parser.add_argument('--train_batch_size', type=int, default=16)
# parser.add_argument('--train_batch_size', type=int, default=8) # try debugging
# parser.add_argument('--display_step',type=int, default=2000)
parser.add_argument('--val_batch_size', type=int, default=2)
parser.add_argument('--test_batch_size', type=int, default=1)
# Model hyperparameters
parser.add_argument('--model_name', type=str, default='facebook/bart-large')
# Optimizer hyperparameters
parser.add_argument('--init_lr', type=float, default=3e-6)
parser.add_argument('--warm_up', type=int, default=600)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--decay_epoch', type=int, default=0)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)
parser.add_argument('--adam_eps', type=float, default=1e-12)
parser.add_argument('--dropout_rate', type=float, default=0.1)
# Tokenizer hyperparameters
parser.add_argument('--encoder_max_len', type=int, default=1024)
parser.add_argument('--decoder_max_len', type=int, default=100)
parser.add_argument('--vocab_size', type=int, default=51201)
parser.add_argument('--eos_idx', type=int, default=51200)
parser.add_argument('--tokenizer_name', type=str, default='RobertaTokenizer')
# Checkpoint directory hyperparameters
parser.add_argument('--pretrained_weight_path', type=str, default='pretrained_weights')
parser.add_argument('--finetune_weight_path', type=str, default="./context_BART_weights_Samsum_5epoch")
parser.add_argument('--best_finetune_weight_path', type=str, default='context_final_BART_weights_Samsum_5epoch')
# Dataset hyperparameters
parser.add_argument('--dataset_name', type=str, default='samsum')
parser.add_argument('--use_paracomet', type=bool, default=False)
parser.add_argument('--use_roberta', type=bool, default=False)
parser.add_argument('--use_sentence_transformer', type=bool, default=False)
parser.add_argument('--dataset_directory', type=str, default='./data')
parser.add_argument('--test_output_file_name', type=str, default='samsum_context_trial2.txt')
parser.add_argument('--relation', type=str, default="xReason")
parser.add_argument('--supervision_relation', type=str, default='isAfter')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("ml6team/keyphrase-extraction-kbir-inspec")

special_tokens_dict = {'additional_special_tokens': ['<I>', '</I>']}
tokenizer.add_special_tokens(special_tokens_dict)

from datasets import load_dataset

dataset_train = load_dataset("samsum", split="train")
dataset_eval = load_dataset("samsum", split="eval")


# Set GPU
print('######################################################################')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('######################################################################')

print('######################################################################')
print('Training Dataset Size is : ')
print(len(dataset_train))
print('Validation Dataset Size is : ')
print(len(dataset_eval))
print('######################################################################')


finetune_model = AutoModelForTokenClassification.from_pretrained("ml6team/keyphrase-extraction-kbir-inspec")
print('######################################################################')
print("Number of Model Parameters are : ", finetune_model.num_parameters())
print('######################################################################')
# Set Training Arguments

finetune_args = Seq2SeqTrainingArguments(
    output_dir=args.finetune_weight_path,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    evaluation_strategy='epoch',
    logging_strategy="epoch",
    save_strategy="epoch",
    # eval_steps=1,
    # logging_steps=1,
    # save_steps=1,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.val_batch_size,
    learning_rate=args.init_lr,
    weight_decay=args.weight_decay,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    adam_epsilon=args.adam_eps,
    num_train_epochs=args.epoch,
    max_grad_norm=0.1,
    # label_smoothing_factor=0.1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    # max_steps= ,
    lr_scheduler_type='polynomial',
    # warmup_ratio= ,
    warmup_steps=args.warm_up,
    save_total_limit=1,
   # fp16=True,
    seed=516,
    load_best_model_at_end=True,
    predict_with_generate=True,
    prediction_loss_only=False,
    generation_max_length=100,
    generation_num_beams=5,
    metric_for_best_model='eval_rouge1',
    greater_is_better=True,
    #  report_to = 'wandb',
)
def compute_metrics(p):
    return_entity_level_metrics = False
    ignore_value = -100
    predictions, labels = p
    label_to_id = {"B": 0, "I": 1, "O": 2}
    id_to_label = ["B", "I", "O"]
    # if model_args.use_crf is False:
    predictions = np.argmax(predictions, axis=2)
    # print(predictions.shape, labels.shape)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != ignore_value]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != ignore_value]
        for prediction, label in zip(predictions, labels)
    ]

    # results = metric.compute(predictions=true_predictions, references=true_labels)
    results = {}
    # print("cal precisi")
    # mode="strict"
    results["overall_precision"] = precision_score(
        true_labels, true_predictions, scheme=IOB2
    )
    results["overall_recall"] = recall_score(true_labels, true_predictions, scheme=IOB2)
    # print("cal f1")
    results["overall_f1"] = f1_score(true_labels, true_predictions, scheme=IOB2)
    results["overall_accuracy"] = accuracy_score(true_labels, true_predictions)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        # print("cal entity level mat")
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

finetune_trainer = Seq2SeqTrainer(
    model=finetune_model,
    args=finetune_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Run Training (Finetuning)
finetune_trainer.train()

# Save final weights
finetune_trainer.save_model(args.best_finetune_weight_path)