
DATA=/usr0/home/yuncheng/MultiBench/synthetic/DATA_redundancy.pickle

python synthetic/early_fusion.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/redundancy_early_fusion_best.pt --modalities 1 1 > synthetic/redundancy_early_fusion.txt
python synthetic/late_fusion.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/redundancy_late_fusion_best.pt --modalities 1 1 > synthetic/redundancy_late_fusion.txt
python synthetic/lrf.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/redundancy_lrf_best.pt --modalities 1 1 > synthetic/redundancy_lrf.txt
python synthetic/tf.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/redundancy_tf_best.pt --modalities 1 1 > synthetic/redundancy_tf.txt
python synthetic/DCCA.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/redundancy_DCCA_best.pt --modalities 1 1 > synthetic/redundancy_DCCA.txt
python synthetic/InfoNCECoordination.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/redundancy_infonce_best.pt --modalities 1 1 > synthetic/redundancy_infonce.txt
python synthetic/mctn.py --data-path ${DATA} --keys 0 1 label --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model /home/yuncheng/redundancy_mctn_best.pt --modalities 1 1 > synthetic/redundancy_mctn.txt
