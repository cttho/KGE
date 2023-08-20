mkdir log
mkdir torch_saved

python main.py --data=kinship --epoch=100 --gpu=0 --model=distmult --embed_dim=400