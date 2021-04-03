CUDA_VISIBLE_DEVICES=3 python main.py --train --config_path=experiments/unfreeze_word_layers.cfg --save_best_model  --resplit_style=speaker_or_utterance_closed --utility --noBLEU --seed 0 --restart
