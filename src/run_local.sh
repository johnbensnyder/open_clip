torchrun --nproc_per_node 8 \
    sm_train.py \
    --train-data "pipe:aws s3 cp s3://jbsnyder-sagemaker-us-west-2/data/laion/laion400m/data/{00000..02499}.tar -" \
    --train-num-samples 1280000 \
    --val-data "pipe:aws s3 cp s3://jbsnyder-sagemaker-us-west-2/data/laion/laion400m/data/{02500..02587}.tar -" \
    --val-num-samples 1000 \
    --batch-size 16 \
    --warmup 250 \
    --epochs 5 \
    --lr 1e-5 \
    --wd 0.2 \
    --precision amp_bfloat16 \
    --workers 8 \
    --log-every-n-steps 10 \
    --log-path ./training_logs \
    --dist-backend nccl