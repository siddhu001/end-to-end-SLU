#!/usr/bin/env bash
#SBATCH --time=08:00:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=4
#SBATCH --err=/jet/home/vviswan3/job_logs/5/seed_1111/single_run.stderr
#SBATCH --output=/jet/home/vviswan3/job_logs/5/seed_1111/single_run.stdout
#SBATCH --mail-type=END
#SBATCH --mail-user=vijayv@andrew.cmu.edu
#SBATCH -p GPU-shared
#SBATCH --gpus=1

training_frac=$1
training_frac_str=$2
seed=$3

# Train and test on FSC splits.
log_suffix=${training_frac_str}_pct_seed_${seed}

echo "Generating (training log) log_${log_suffix}.csv"
python main.py --train --config_path=experiments/no_unfreezing.cfg \
--training_fraction $training_frac --log_suffix original --seed ${seed}

# Test on the Snips test set.
log_suffix=snips_${training_frac_str}_pct_seed_${seed}

python test.py --restart --model_path model_state_${training_frac_str}_pct_seed_${seed}.pth \
    --config_path=experiments/no_unfreezing_snips_test.cfg --error_path error_analysis.csv \
    --snips_test_set --log_file_suffix ${log_suffix}


# Train and test on FSC splits (testing on speaker-closed split).
log_suffix=spk_or_utt_closed_with_utility_spk_test_${training_frac_str}_pct_seed_${seed}

echo "Generating (training log) log_${log_suffix}.csv"
python main.py --train --speaker_or_utterance_closed_with_utility_split --config_path=experiments/no_unfreezing.cfg \
--training_fraction $training_frac --seed ${seed}


# Test on the FSC utterance-closed test split.
log_suffix=spk_or_utt_closed_with_utility_utt_test_${training_frac_str}_pct_seed_${seed}

echo "Generating (test log) log_${log_suffix}.csv"
python test.py --restart --model_path model_state_spk_or_utt_closed_with_utility_spk_test_${training_frac_str}_pct_seed_${seed}.pth \
--speaker_or_utterance_closed_with_utility_utterance_test --config_path=experiments/no_unfreezing_snips_test.cfg \
--error_path error_analysis.csv \
--log_file_suffix ${log_suffix}


# Test on the Snips test set.
log_suffix=spk_or_utt_closed_with_utility_trained_snips_${training_frac_str}_pct_seed_${seed}

echo "Generating (test log) log_${log_suffix}.csv"
python test.py --restart --model_path model_state_spk_or_utt_closed_with_utility_spk_test_${training_frac_str}_pct_seed_${seed}.pth \
--config_path=experiments/no_unfreezing_snips_test.cfg --error_path error_analysis.csv --snips_test_set \
--log_file_suffix ${log_suffix}