
for ((i=1; i<=5; i++))
do
    python synthetic/generate_data.py --num-data 20000 --setting synthetic$i --out-path synthetic/model_selection
done
