#!/usr/bin/env bash
if [ $# -eq 0 ]
  then
    num_models=16
else
    num_models=$1
fi
max_iter=$(expr $num_models - 1)
seed=3333

training_fractions=( 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 )
training_frac_strings=( 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 )
num_models=20


printf "###########\nTraining and testing on original splits\n####\n"
# Training and testing on original splits.
for ((i=0; i<=$max_iter; i++))
do
    training_frac=${training_fractions[i]}

    training_frac_str=${training_frac_strings[i]}
    echo "Running evaluation for training frac ${training_frac_str}"

    log_suffix=snips_${training_frac_str}_pct_seed_${seed}
    python test.py --restart --model_path model_state_${training_frac_str}_pct_seed_${seed}.pth \
        --config_path=experiments/no_unfreezing_snips_test.cfg --error_path error_analysis.csv \
        --snips_test_set --log_file_suffix ${log_suffix}


    # Test on the FSC utterance-closed test split.
    log_suffix=spk_or_utt_closed_with_utility_utt_test_noBLEU_${training_frac_str}_pct_seed_${seed}

    echo "Generating (test log) log_${log_suffix}.csv"
    python test.py --restart --model_path model_state_spk_or_utt_closed_with_utility_spk_test_noBLEU_${training_frac_str}_pct_seed_${seed}.pth \
    --speaker_or_utterance_closed_with_utility_utterance_test --config_path=experiments/no_unfreezing_snips_test.cfg \
    --error_path error_analysis.csv \
    --log_file_suffix ${log_suffix}


    # Test on the Snips test set.
    log_suffix=spk_or_utt_closed_with_utility_trained_snips_${training_frac_str}_pct_seed_${seed}

    echo "Generating (test log) log_${log_suffix}.csv"
    python test.py --restart --model_path model_state_spk_or_utt_closed_with_utility_spk_test_noBLEU_${training_frac_str}_pct_seed_${seed}.pth \
    --config_path=experiments/no_unfreezing_snips_test.cfg --error_path error_analysis.csv --snips_test_set \
    --log_file_suffix ${log_suffix}

    printf "\n"
done

printf "###########\nDone."