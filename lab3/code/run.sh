models=('deepconvnet')
batches=(64) # 64
lrs=(5e-4) # 5e-4
optims=('adam') # adam
num_epochs=500
dropout=0.75 # 0.5
gpu=3

for model in ${models[@]}
do
    for batch in ${batches[@]}
    do
        for lr in ${lrs[@]}
        do
            for optim in ${optims[@]}
            do
                python main.py \
                    --model $model \
                    --batch_size $batch \
                    --lr $lr \
                    --optim $optim \
                    --num_epochs $num_epochs \
                    --dropout $dropout \
                    --gpu $gpu \
                    --epoch_log $num_epochs
            done
        done
    done
done