
for setting in redundancy uniqueness0 uniqueness1 synergy mix1 mix2 mix3 mix4 mix5 mix6 synthetic1 synthetic2 synthetic3 synthetic4 synthetic5
do
    echo ${setting}
    echo base
    python synthetic/experiments/base.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --n-latent 600 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_base_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_base.txt

    echo agree
    python synthetic/experiments/agree.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --n-latent 600 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_agree_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_agree.txt

    echo align
    python synthetic/experiments/align.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --n-latent 600 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_align_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_align.txt

    echo recon
    python synthetic/experiments/recon.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --n-latent 600 --saved-model synthetic/experiments/${setting}/${setting}_recon_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_recon.txt

    echo mctn
    python synthetic/mctn.py  --data-path synthetic/experiments/DATA_${setting}.pickle --key 0 1 label --bs 256 --input-dim 200 --hidden-dim 200 --output-dim 600 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_mctn_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_mctn.txt

    echo early_fusion
    python synthetic/early_fusion.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_early_fusion_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_early_fusion.txt

    echo lrf
    python synthetic/lrf.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 600 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_lower_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_lower.txt

    echo outer
    python synthetic/outer.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 600 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_outer_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_outer.txt

    echo additive
    python synthetic/additive.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 600 --hidden-dim 20 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_additive_best.pt --setting ${setting}  > synthetic/experiments/${setting}/${setting}_additive.txt

    echo elem
    python synthetic/elem.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 600 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_elem_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_elem.txt

    echo mult 
    python synthetic/mult.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_mult_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_mult.txt

    echo mi
    python synthetic/mi.py --data-path synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --output-dim 600 --num-classes 2 --saved-model synthetic/experiments/${setting}/${setting}_mi_best.pt --modalities 1 1 > synthetic/experiments/${setting}/${setting}_mi.txt
done
