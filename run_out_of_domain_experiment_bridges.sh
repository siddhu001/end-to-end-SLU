#!/usr/bin/env bash
if [ $# -eq 0 ]
  then
    num_models=20
else
    num_models=$1
fi
max_iter=$(expr $num_models - 1)

training_fractions=( 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 )
training_frac_str=( 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 )

seeds=( 1111 2222 3333 4444 5555 )


training_fractions=( 0.05 )
training_frac_str=( 5 )

num_models=1
seeds=( 1111 )

printf "###########\nTraining and testing on original splits\n####\n"
# Training and testing on original splits.
for ((i=0; i<=$max_iter; i++))
do
    for seed in ${seeds[@]}; do
        training_frac=${training_fractions[i]}
        training_frac_str=${training_frac_str[i]}
        echo "Launching jobs for ${training_frac_str} percent of data."

        sbatch run_single_out_of_domain_bridges.sh ${training_frac} ${training_frac_str} ${seed}

        printf "\n"
    done
done

printf "###########\nDone."


