# OpenPath
[MICCAI 2025] Official code for OpenPath

## Code usage
```
python attempts/vlm_crc100k_ood_random.py --model_type BMC --init_num 50 --save_csv al_file/BMC_openpath_init.csv \
    --seed 2020 --ood_cls 0 1 2 4 5 7 
```

```
python train_sup_crc100k.py --log log/train_sup_lit.txt --epochs 50 --val_epochs 1 \
        --batch_size 128 --lr 1e-3 --query_num 50 --seed $param \
        --init_csv al_file/BMC_openpath_init.csv --id_cls 3 6 8 --ood_cls 0 1 2 4 5 7 --id_ratio 0.25 --query_times 5 \
        --dataset CRC
```
