
for ((i=1; i<=5; i++))
do
    echo synthetic${i}
    echo early_fusion
    python synthetic/early_fusion.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --input-dim 200 --hidden-dim 20 --num-classes 2 --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_early_fusion_best.pt --modalities 1 1 > synthetic/model_selection/synthetic${i}/synthetic${i}_early_fusion.txt
    echo lrf
    python synthetic/lrf.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 512 --hidden-dim 20 --num-classes 2 --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_lower_best.pt --modalities 1 1 > synthetic/model_selection/synthetic${i}/synthetic${i}_lower.txt
    echo outer
    python synthetic/outer.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 512 --hidden-dim 20 --num-classes 2 --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_outer_best.pt --modalities 1 1 > synthetic/model_selection/synthetic${i}/synthetic${i}_outer.txt
    echo additive
    python synthetic/additive.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 512 --hidden-dim 20 --num-classes 2 --setting synthetic${i} --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_additive_best.pt > synthetic/model_selection/synthetic${i}/synthetic${i}_additive.txt
    echo mfm
    python synthetic/mfm.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_mfm_best.pt --modalities 1 1 > synthetic/model_selection/synthetic${i}/synthetic${i}_mfm.txt
    echo elem
    python synthetic/elem.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_elem_best.pt --modalities 1 1 > synthetic/model_selection/synthetic${i}/synthetic${i}_elem.txt
    echo agree
    python synthetic/agree.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --setting synthetic${i} --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_agree_best.pt > synthetic/model_selection/synthetic${i}/synthetic${i}_agree.txt
    python synthetic/mult.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_mult_best.pt --modalities 1 1 > synthetic/model_selection/synthetic${i}/synthetic${i}_mult.txt
    echo mi
    python synthetic/mi.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --output-dim 512 --num-classes 2 --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_mi_best.pt --modalities 1 1 > synthetic/model_selection/synthetic${i}/synthetic${i}_mi.txt
    echo align
    python synthetic/align.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --setting synthetic${i} --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_align_best.pt > synthetic/model_selection/synthetic${i}/synthetic${i}_align.txt
done

for ((i=1; i<=5; i++))
do
    python synthetic/unimodal.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_unimodal0 --modality 0 > synthetic/model_selection/synthetic${i}/synthetic${i}_unimodal0.txt
    python synthetic/unimodal.py --data-path synthetic/model_selection/DATA_synthetic${i}.pickle --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model synthetic/model_selection/synthetic${i}/synthetic${i}_unimodal1 --modality 1 > synthetic/model_selection/synthetic${i}/synthetic${i}_unimodal1.txt
done


python synthetic/unimodal.py --data-path synthetic/model_selection/DATA_maps.pickle --keys 0 1 label --input-dim 1000 --output-dim 512 --hidden-dim 512 --num-classes 3 --saved-model synthetic/model_selection/maps/maps_unimodal0 --modality 0 > synthetic/model_selection/maps/maps_unimodal0.txt
python synthetic/unimodal.py --data-path synthetic/model_selection/DATA_maps.pickle --keys 0 1 label --input-dim 137 --output-dim 512 --hidden-dim 512 --num-classes 3 --saved-model synthetic/model_selection/maps/maps_unimodal1 --modality 1 > synthetic/model_selection/maps/maps_unimodal1.txt

python synthetic/early_fusion.py --data-path synthetic/model_selection/DATA_maps2.pickle --keys 0 1 label --input-dim 1000 18 --hidden-dim 512 --num-classes 3 --saved-model synthetic/model_selection/maps2/maps2_early_fusion_best.pt --modalities 1 1 > synthetic/model_selection/maps2/maps2_early_fusion.txt
python synthetic/lrf.py --data-path synthetic/model_selection/DATA_maps2.pickle --keys 0 1 label --input-dim 1000 18 --output-dim 512 --hidden-dim 20 --num-classes 3 --saved-model synthetic/model_selection/maps2/maps2_lower_best.pt --modalities 1 1 > synthetic/model_selection/maps2/maps2_lower.txt
python synthetic/outer.py --data-path synthetic/model_selection/DATA_maps2.pickle --keys 0 1 label --input-dim 1000 18 --output-dim 512 --hidden-dim 20 --num-classes 3 --saved-model synthetic/model_selection/maps2/maps2_outer_best.pt --modalities 1 1 > synthetic/model_selection/maps2/maps2_outer.txt
python synthetic/additive.py --data-path synthetic/model_selection/DATA_maps2.pickle --keys 0 1 label --input-dim 1000 18 --output-dim 512 --hidden-dim 512 --num-classes 3 --setting maps2 --saved-model synthetic/model_selection/maps2/maps2_additive_best.pt > synthetic/model_selection/maps2/maps2_additive.txt
python synthetic/mfm.py --data-path synthetic/model_selection/DATA_maps2.pickle --keys 0 1 label --input-dim 1000 18 --hidden-dim 512 --num-classes 3 --saved-model synthetic/model_selection/maps2/maps2_mfm_best.pt --modalities 1 1 > synthetic/model_selection/maps2/maps2_mfm.txt
python synthetic/elem.py --data-path synthetic/model_selection/DATA_maps2.pickle --keys 0 1 label --input-dim 1000 18 --hidden-dim 512 --num-classes 3 --saved-model synthetic/model_selection/maps2/maps2_elem_best.pt --modalities 1 1 > synthetic/model_selection/maps2/maps2_elem.txt
python synthetic/agree.py --data-path synthetic/model_selection/DATA_maps2.pickle --keys 0 1 label --input-dim 1000 18 --output-dim 512 --hidden-dim 512 --num-classes 3 --setting maps2 --saved-model synthetic/model_selection/maps2/maps2_agree_best.pt > synthetic/model_selection/maps2/maps2_agree.txt
python synthetic/mult.py --data-path synthetic/model_selection/DATA_maps2.pickle --keys 0 1 label --input-dim 1000 18 --hidden-dim 512 --num-classes 3 --saved-model synthetic/model_selection/maps2/maps2_mult_best.pt --modalities 1 1 > synthetic/model_selection/maps2/maps2_mult.txt
python synthetic/mi.py --data-path synthetic/model_selection/DATA_maps2.pickle --keys 0 1 label --input-dim 1000 18 --hidden-dim 512 --output-dim 512 --num-classes 3 --saved-model synthetic/model_selection/maps2/maps2_mi_best.pt --modalities 1 1 > synthetic/model_selection/maps2/maps2_mi.txt
python synthetic/align.py --data-path synthetic/model_selection/DATA_maps2.pickle --keys 0 1 label --input-dim 1000 18 --output-dim 512 --hidden-dim 512 --num-classes 3 --setting maps2 --saved-model synthetic/model_selection/maps2/maps2_align_best.pt > synthetic/model_selection/maps2/maps2_align.txt
