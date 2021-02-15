if [ $# -eq 0 ]
  then
    num_models=20
else
    num_models=$1
fi
max_iter=$(expr $num_models - 1)

training_fractions=( 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 )
training_frac_str=( 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 )

printf "###########\nTraining and testing on original splits\n####\n"
# Training and testing on original splits.
for i in {0..$max_iter}
do
    training_frac=${training_fractions[i]}
    training_frac_str=${training_frac_str[i]}


    # Train and test on FSC splits.
    log_suffix=${training_frac_str}_pct

    printf "Generating (training log) log_${log_suffix}.csv"
    python main.py --train --config_path=experiments/no_unfreezing.cfg --training_fraction $training_frac


    # Test on the Snips test set.
    log_suffix=snips_${training_frac_str}_pct

    python test.py --restart --model_path model_state_${training_frac_str}_pct.pth \
     --config_path=experiments/no_unfreezing_snips_test.cfg --error_path error_analysis.csv \
     --snips_test_set --log_file_suffix ${log_suffix}

    printf "\n"
done

printf "\n\n\n####\nTraining and testing on speaker-or-utterance-closed splits\n####\n"
# Training on speaker-or-utterance-closed splits, and testing on speaker-closed test split.
for i in {0..$max_iter}
do
    training_frac=${training_fractions[i]}
    training_frac_str=${training_frac_str[i]}


    # Train and test on FSC splits (testing on speaker-closed split).
    log_suffix=spk_or_utt_closed_spk_test_${training_frac_str}_pct

    printf "Generating (training log) log_${log_suffix}.csv"
    python main.py --train --speaker_or_utterance_closed_split --config_path=experiments/no_unfreezing.cfg --training_fraction $training_frac


    # Test on the FSC utterance-closed test split.
    log_suffix=spk_or_utt_closed_utt_test_${training_frac_str}_pct

    printf "Generating (test log) log_${log_suffix}.csv"
    python test.py --restart --model_path model_state_spk_or_utt_closed_${training_frac_str}_pct.pth \
    --speaker_or_utterance_closed_utterance_test --config_path=experiments/no_unfreezing_snips_test.cfg \
    --error_path error_analysis.csv --snips_test_set \
    --log_file_suffix ${log_suffix}


    # Test on the Snips test set.
    log_suffix=spk_or_utt_closed_trained_snips_${training_frac_str}_pct

    printf "Generating (test log) log_${log_suffix}.csv"
    python test.py --restart --model_path model_state_spk_or_utt_closed_${training_frac_str}_pct.pth \
    --config_path=experiments/no_unfreezing_snips_test.cfg --error_path error_analysis.csv --snips_test_set \
    --log_file_suffix ${log_suffix}

    printf "\n"
done
