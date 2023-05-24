
for SETTING in redundancy uniqueness0 uniqueness1 synergy
do
    echo ${SETTING}
    echo early_fusion
    python synthetic/early_fusion.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --input-dim 200 --hidden-dim 20 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_early_fusion_best.pt --modalities 1 1 > synthetic/experiments/${SETTING}/${SETTING}_early_fusion.txt
    echo lrf
    python synthetic/lrf.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 20 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_lower_best.pt --modalities 1 1 > synthetic/experiments/${SETTING}/${SETTING}_lower.txt
    echo outer
    python synthetic/outer.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 20 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_outer_best.pt --modalities 1 1 > synthetic/experiments/${SETTING}/${SETTING}_outer.txt
    echo additive
    python synthetic/additive.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 512 --hidden-dim 20 --num-classes 2 --setting ${SETTING} --saved-model synthetic/experiments/${SETTING}/${SETTING}_additive_best.pt > synthetic/experiments/${SETTING}/${SETTING}_additive.txt
    echo mfm
    python synthetic/mfm.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_mfm_best.pt --modalities 1 1 > synthetic/experiments/${SETTING}/${SETTING}_mfm.txt
    echo elem
    python synthetic/elem.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_elem_best.pt --modalities 1 1 > synthetic/experiments/${SETTING}/${SETTING}_elem.txt
    echo elem_recon
    python synthetic/elem_recon.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_elem_recon_best.pt --modalities 1 1 > synthetic/experiments/${SETTING}/${SETTING}_elem_recon.txt
    echo agree
    python synthetic/agree.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --setting ${SETTING} --saved-model synthetic/experiments/${SETTING}/${SETTING}_agree_best.pt > synthetic/experiments/${SETTING}/${SETTING}_agree.txt
    echo mult 
    python synthetic/mult.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_mult_best.pt --modalities 1 1 > synthetic/experiments/${SETTING}/${SETTING}_mult.txt
    echo mi
    python synthetic/mi.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --output-dim 512 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_mi_best.pt --modalities 1 1 > synthetic/experiments/${SETTING}/${SETTING}_mi.txt
    echo mi_recon
    python synthetic/mi_recon.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_mi_recon_best.pt --modalities 1 1 > synthetic/experiments/${SETTING}/${SETTING}_mi_recon.txt
    echo align
    python synthetic/align.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --setting ${SETTING} --saved-model synthetic/experiments/${SETTING}/${SETTING}_align_best.pt > synthetic/experiments/${SETTING}/${SETTING}_align.txt
done

for SETTING in redundancy uniqueness0 uniqueness1 synergy mix1 mix2 mix3 mix4 mix5 mix6
do
    python synthetic/unimodal.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_unimodal0 --modality 0 > synthetic/experiments/${SETTING}/${SETTING}_unimodal0.txt
    python synthetic/unimodal.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_unimodal1 --modality 1 > synthetic/experiments/${SETTING}/${SETTING}_unimodal1.txt
done
