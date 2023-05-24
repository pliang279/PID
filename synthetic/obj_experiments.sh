
for W in 0.1 0.2 0.5
do 
    for SETTING in redundancy uniqueness0 uniqueness1 synergy mix1 mix2 mix3 mix4 mix5 mix6
    do
        python synthetic/agree.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 20 --num-classes 2 --setting ${SETTING} --weight ${W} --saved-model synthetic/experiments/${SETTING}/${SETTING}_agree${W}_best.pt > synthetic/experiments/${SETTING}/${SETTING}_agree${W}.txt
        python synthetic/align.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --input-dim 200 --output-dim 512 --hidden-dim 20 --num-classes 2 --setting ${SETTING} --weight ${W} --saved-model synthetic/experiments/${SETTING}/${SETTING}_align${W}_best.pt > synthetic/experiments/${SETTING}/${SETTING}_align${W}.txt
        python synthetic/mfm.py --data-path synthetic/DATA_${SETTING}.pickle --keys 0 1 label --input-dim 200 --hidden-dim 512 --num-classes 2 --saved-model synthetic/experiments/${SETTING}/${SETTING}_mfm${W}_best.pt --modalities 1 1 --weight ${W} > synthetic/experiments/${SETTING}/${SETTING}_mfm${W}.txt
    done
done

