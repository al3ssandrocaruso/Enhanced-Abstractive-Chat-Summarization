import argparse

import dataset
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification
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

# for now, only samsum
total_dataset = dataset.SamsumDataset_total(args.encoder_max_len, args.decoder_max_len, tokenizer, extra_context=True,
                                                paracomet=args.use_paracomet, relation=args.relation,
                                                supervision_relation=args.supervision_relation, roberta=args.use_roberta,
                                                sentence_transformer=args.use_sentence_transformer)
train_dataset = total_dataset.getTrainData()
eval_dataset = total_dataset.getEvalData()
test_dataset = total_dataset.getTestData()

# Set GPU
print('######################################################################')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('######################################################################')

print('######################################################################')
print('Training Dataset Size is : ')
print(len(train_dataset))
print('Validation Dataset Size is : ')
print(len(eval_dataset))
print('Test Dataset Size is : ')
print(len(test_dataset))
print('######################################################################')


finetune_model = AutoModelForTokenClassification.from_pretrained("ml6team/keyphrase-extraction-kbir-inspec")
print('######################################################################')
print("Number of Model Parameters are : ", finetune_model.num_parameters())
print('######################################################################')
