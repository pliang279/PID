
python synthetic/generate_data.py --num-data 20000 --setting redundancy --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2
python synthetic/generate_data.py --num-data 20000 --setting uniqueness0 --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2
python synthetic/generate_data.py --num-data 20000 --setting uniqueness1 --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2
python synthetic/generate_data.py --num-data 20000 --setting synergy --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2
python synthetic/generate_data.py --num-data 20000 --setting mix1 --mix-ratio 0.5 0.5 0.5 --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2
python synthetic/generate_data.py --num-data 20000 --setting mix2 --mix-ratio 1 0.5 0.5 --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2
python synthetic/generate_data.py --num-data 20000 --setting mix3 --mix-ratio 1 1 0.5 --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2
python synthetic/generate_data.py --num-data 20000 --setting mix4 --mix-ratio 0.5 0.5 1 --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2
python synthetic/generate_data.py --num-data 20000 --setting mix5 --mix-ratio 0 0.5 0.5 --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2
python synthetic/generate_data.py --num-data 20000 --setting mix6 --mix-ratio 0 0.5 1 --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2

for ((i=1; i<=5; i++))
do
    python synthetic/generate_data.py --num-data 20000 --setting synthetic$i --out-path /usr0/home/yuncheng/MultiBench/synthetic/experiments2
done
