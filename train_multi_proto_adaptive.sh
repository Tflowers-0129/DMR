python main.py mydataset RGB Flow STFT STFT_2 --config ./exps/aid_multiproto_adaptive.json \
    --train_list mydataset_train.txt --val_list mydataset_test.txt \
    --mpu_path '/home/amax/Downloads/whx/temporal-binding-network/dataset/gyro/' \
    --arch BNInception --num_segments 8 --dropout 0.5  --epochs 20 -b 8 --lr 0.001 \
    --lr_steps 10 --gd 50 --partialbn -j 8 --pretrained_flow ./pretrain_model/TSN-kinetics-flow.pth