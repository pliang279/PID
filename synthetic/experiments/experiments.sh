
for setting in redundancy uniqueness0 uniqueness1 synergy mix1 mix2 mix3 mix4 mix5 mix6 synthetic1 synthetic2 synthetic3 synthetic4 synthetic5
do
    echo ${setting}
    # echo base
    # python synthetic/experiments2/base.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --n-latent 600 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_base_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_base.txt

    # echo agree
    # python synthetic/experiments2/agree.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --n-latent 600 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_agree_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_agree.txt

    # echo align
    # python synthetic/experiments2/align.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --n-latent 600 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_align_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_align.txt

    # echo recon
    # python synthetic/experiments/recon.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --n-latent 600 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_recon_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_recon.txt

    # echo mctn
    # python synthetic/mctn.py  --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --key 0 1 label --bs 256 --input-dim 200 --hidden-dim 200 --output-dim 600 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_mctn_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_mctn.txt

    # echo early_fusion
    # python synthetic/early_fusion.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_early_fusion_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_early_fusion.txt

    # echo lrf
    # python synthetic/lrf.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 600 --hidden-dim 512 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_lower_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_lower.txt

    # echo outer
    # python synthetic/outer.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 600 --hidden-dim 512 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_outer_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_outer.txt

    echo additive
    python synthetic/additive.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 600 --hidden-dim 20 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_additive_best.pt --setting ${setting}  > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_additive.txt

    # echo elem
    # python synthetic/elem.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --output-dim 600 --hidden-dim 512 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_elem_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_elem.txt

    # echo mult 
    # python synthetic/mult.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_mult_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_mult.txt

    # echo mi
    # python synthetic/mi.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_${setting}.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --output-dim 600 --num-classes 2 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_mi_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/${setting}/${setting}_mi.txt
done


echo maps
# echo base
# python synthetic/experiments2/base.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --keys 0 1 label --bs 256 --input-dim 1000 137 --hidden-dim 512 --n-latent 600 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_base_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_base.txt

# echo agree
# python synthetic/experiments2/agree.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --keys 0 1 label --bs 256 --input-dim 1000 137 --hidden-dim 512 --n-latent 600 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_agree_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_agree.txt

# echo align
# python synthetic/experiments2/align.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --keys 0 1 label --bs 256 --input-dim 1000 137 --hidden-dim 512 --n-latent 600 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_align_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_align.txt

# echo recon
# python synthetic/mfm.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --keys 0 1 label --bs 256 --input-dim 1000 137 --hidden-dim 512 --num-classes 3 --n-latent 600 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_recon_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_recon.txt

# echo mctn
# python synthetic/mctn.py  --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --key 0 1 label --bs 256 --input-dim 1000 137 --hidden-dim 512 --output-dim 600 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_mctn_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_mctn.txt

# echo early_fusion
# python synthetic/early_fusion.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --keys 0 1 label --bs 256 --input-dim 1000 137 --hidden-dim 512 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_early_fusion_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_early_fusion.txt

# echo lrf
# python synthetic/lrf.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --keys 0 1 label --bs 256 --input-dim 1000 137 --output-dim 600 --hidden-dim 512 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_lower_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_lower.txt

# echo outer
# python synthetic/outer.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --keys 0 1 label --bs 256 --input-dim 1000 137 --output-dim 600 --hidden-dim 512 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_outer_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_outer.txt

echo additive
python synthetic/additive.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --setting maps --keys 0 1 label --bs 256 --input-dim 1000 137 --output-dim 600 --hidden-dim 20 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_additive_best.pt > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_additive.txt

# echo elem
# python synthetic/elem.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --keys 0 1 label --bs 256 --input-dim 1000 137 --output-dim 600 --hidden-dim 512 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_elem_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_elem.txt

#     echo mult 
#     python synthetic/mult.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --keys 0 1 label --bs 128 --input-dim 1000 137 --hidden-dim 512 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_mult_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_mult.txt

#     echo mi
#     python synthetic/mi.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps.pickle --keys 0 1 label --bs 128 --input-dim 1000 137 --hidden-dim 512 --output-dim 600 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_mi_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps/maps_mi.txt


echo maps2
# echo base
# python synthetic/experiments2/base.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --keys 0 1 label --bs 256 --input-dim 1000 18 --hidden-dim 512 --n-latent 600 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_base_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_base.txt

# echo agree
# python synthetic/experiments2/agree.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --keys 0 1 label --bs 256 --input-dim 1000 18 --hidden-dim 512 --n-latent 600 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_agree_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_agree.txt

# echo align
# python synthetic/experiments2/align.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --keys 0 1 label --bs 256 --input-dim 1000 18 --hidden-dim 512 --n-latent 600 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_align_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_align.txt

# echo recon
# python synthetic/mfm.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --keys 0 1 label --bs 256 --input-dim 1000 18 --hidden-dim 512 --num-classes 3 --n-latent 600 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_recon_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_recon.txt

# echo mctn
# python synthetic/mctn.py  --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --key 0 1 label --bs 256 --input-dim 1000 18 --hidden-dim 512 --output-dim 600 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_mctn_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_mctn.txt

# echo early_fusion
# python synthetic/early_fusion.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --keys 0 1 label --bs 256 --input-dim 1000 18 --hidden-dim 512 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_early_fusion_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_early_fusion.txt

# echo lrf
# python synthetic/lrf.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --keys 0 1 label --bs 256 --input-dim 1000 18 --output-dim 600 --hidden-dim 512 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_lower_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_lower.txt

# echo outer
# python synthetic/outer.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --keys 0 1 label --bs 256 --input-dim 1000 18 --output-dim 600 --hidden-dim 512 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_outer_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_outer.txt

echo additive
python synthetic/additive.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --setting maps2 --keys 0 1 label --bs 256 --input-dim 1000 18 --output-dim 600 --hidden-dim 20 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_additive_best.pt > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_additive.txt

# echo elem
# python synthetic/elem.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --keys 0 1 label --bs 256 --input-dim 1000 18 --output-dim 600 --hidden-dim 512 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_elem_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_elem.txt

#     echo mult 
#     python synthetic/mult.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --keys 0 1 label --bs 128 --input-dim 1000 18 --hidden-dim 512 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_mult_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_mult.txt

#     echo mi
#     python synthetic/mi.py --data-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2/DATA_maps2.pickle --keys 0 1 label --bs 128 --input-dim 1000 18 --hidden-dim 512 --output-dim 600 --num-classes 3 --saved-model /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_mi_best.pt --modalities 1 1 > /usr0/home/yuncheng/MultiBench/synthetic/experiments2/maps2/maps2_mi.txt
