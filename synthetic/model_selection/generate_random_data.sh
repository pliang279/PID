
for ((i=1; i<=5; i++))
do
    python synthetic/generate_data.py --num-data 20000 --setting synthetic$i --out-path /usr0/home/yuncheng/MultiBench/synthetic/model_selection
done
