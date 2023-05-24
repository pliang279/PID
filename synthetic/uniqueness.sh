
# DATA=synthetic/DATA_uniqueness0.pickle

# python synthetic/early_fusion.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness0_early_fusion_best.pt --modalities 1 1 > synthetic/uniqueness0_early_fusion.txt
# python synthetic/late_fusion.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness0_late_fusion_best.pt --modalities 1 1 > synthetic/uniqueness0_late_fusion.txt
# python synthetic/lrf.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness0_lrf_best.pt --modalities 1 1 > synthetic/uniqueness0_lrf.txt
# python synthetic/tf.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness0_tf_best.pt --modalities 1 1 > synthetic/uniqueness0_tf.txt
# python synthetic/DCCA.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness0_DCCA_best.pt --modalities 1 1 > synthetic/uniqueness0_DCCA.txt
# python synthetic/InfoNCECoordination.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness0_infonce_best.pt --modalities 1 1 > synthetic/uniqueness0_infonce.txt
# python synthetic/mctn.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness0_mctn_best.pt --modalities 1 1 > synthetic/uniqueness0_mctn.txt


DATA=synthetic/DATA_uniqueness1.pickle

python synthetic/early_fusion.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness1_early_fusion_best.pt --modalities 1 1 > synthetic/uniqueness1_early_fusion.txt
# python synthetic/late_fusion.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness1_late_fusion_best.pt --modalities 1 1 > synthetic/uniqueness1_late_fusion.txt
# python synthetic/lrf.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness1_lrf_best.pt --modalities 1 1 > synthetic/uniqueness1_lrf.txt
python synthetic/tf.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness1_tf_best.pt --modalities 1 1 > synthetic/uniqueness1_tf.txt
python synthetic/DCCA.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness1_DCCA_best.pt --modalities 1 1 > synthetic/uniqueness1_DCCA.txt
# python synthetic/InfoNCECoordination.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness1_infonce_best.pt --modalities 1 1 > synthetic/uniqueness1_infonce.txt
# python synthetic/mctn.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/uniqueness1_mctn_best.pt --modalities 1 1 > synthetic/uniqueness1_mctn.txt
