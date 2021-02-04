# Create error analysis file for model
import torch
import numpy as np
from models import PretrainedModel, Model, obtain_fasttext_embeddings
from data import get_ASR_datasets, get_SLU_datasets, read_config
from training import Trainer
import argparse
import os
if __name__ == '__main__':
	# Get args
	parser = argparse.ArgumentParser()
	parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
	parser.add_argument('--config_path', type=str, required=True, help='path to config file with hyperparameters, etc.')
	parser.add_argument('--error_path', type=str, required=True, help='path to store list of files with predicted errors.')
	parser.add_argument('--model_path', type=str, required=True, help='path of model to load')
	parser.add_argument('--use_FastText_embeddings', action='store_true', help='use FastText embeddings')
	parser.add_argument('--semantic_embeddings_path', type=str, help='path for semantic embeddings')
	parser.add_argument('--disjoint_split', action='store_true', help='split dataset with disjoint utterances in train set and test set')
	parser.add_argument('--utterance_closed_split', action='store_true', help='load closed utterance train/val/test')
	parser.add_argument('--utterance_closed_with_utility_split', action='store_true', help='load closed utterance (utility) train/val/test')
	parser.add_argument('--smooth_semantic', action='store_true', help='sum semantic embedding of top k words')
	parser.add_argument('--smooth_semantic_parameter', type=str, default="5",help='value of k in smooth_smantic')
	parser.add_argument('--nlu_setup', action='store_true', help='use Gold utterances to run an NLU test pipeline')
	parser.add_argument('--asr_setup', action='store_true', help='use Gold utterances to run an ASR test pipeline')
	parser.add_argument('--single_label', action='store_true',help='Whether our dataset contains a single intent label (or a full triple). Only applied for the FSC dataset.')
	
	args = parser.parse_args()
	restart = args.restart
	config_path = args.config_path
	model_path = args.model_path
	use_FastText_embeddings = args.use_FastText_embeddings
	semantic_embeddings_path = args.semantic_embeddings_path
	disjoint_split = args.disjoint_split
	utterance_closed_split = args.utterance_closed_split
	utterance_closed_with_utility_split = args.utterance_closed_with_utility_split
	smooth_semantic = args.smooth_semantic
	smooth_semantic_parameter = int(args.smooth_semantic_parameter)
	nlu_setup = args.nlu_setup
	asr_setup = args.asr_setup
	single_intent = args.single_label

	# Read config file
	config = read_config(config_path)
	torch.manual_seed(config.seed); np.random.seed(config.seed)

	# Generate datasets
	use_gold_utterances = False
	use_all_gold=False
	if nlu_setup:
		use_gold_utterances = True
		use_all_gold=True

	train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config, disjoint_split=disjoint_split, utterance_closed = utterance_closed_split,\
	utterance_closed_with_utility=utterance_closed_with_utility_split, use_all_gold = use_all_gold, use_gold_utterances = use_gold_utterances, single_label=single_intent, asr_setup = asr_setup)

	# Initialize model
	if use_FastText_embeddings:
		# Load FastText Embedding
		Sy_word = []
		with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
			for line in f.readlines():
				Sy_word.append(line.rstrip("\n"))
		FastText_embeddings=obtain_fasttext_embeddings(semantic_embeddings_path, Sy_word)
		model = Model(config=config,pipeline=False, use_semantic_embeddings = use_FastText_embeddings, glove_embeddings=FastText_embeddings,glove_emb_dim=300, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
	else:
		model = Model(config=config)

	# Load the trained model
	trainer = Trainer(model=model, config=config)
	if restart: trainer.load_checkpoint(model_path)
	# Create csv file containing errors made by model
	if not asr_setup:
		test_intent_acc, test_intent_loss = trainer.get_error(test_dataset, error_path=args.error_path, nlu_setup = nlu_setup)
		print("========= Test results =========")
		print("*intents*| test accuracy: %.2f| test loss: %.2f\n" % (test_intent_acc, test_intent_loss) )
	else:
		# note, error path not logging errors yet 
		test_wer = trainer.get_asr_error(test_dataset, error_path=args.error_path)
		print("========= Test results =========")
		print("Average WER (weighted by length) : {:.2f}".format(test_wer))

		
	

