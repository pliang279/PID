
for setting in redundancy uniqueness0 uniqueness1 synergy mix1 mix2 mix3 mix4 mix5 mix6 synthetic1 synthetic2 synthetic3 synthetic4 synthetic5
do
    python synthetic/unimodal.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --input-dim 200 --output-dim 600 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_unimodal0 --modality 0 > synthetic/experiments/${setting}/${setting}_unimodal0.txt
    python synthetic/unimodal.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --input-dim 200 --output-dim 600 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_unimodal1 --modality 1 > synthetic/experiments/${setting}/${setting}_unimodal1.txt
done
