dset_name=tvsum
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_video_tvsum
exp_id=exp


######## data paths
train_path=data/tvsum/tvsum_train.jsonl
eval_path=data/tvsum/tvsum_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=/home/data/yj/DATA/tvsum

# # video features
v_feat_dim=2048
v_feat_dirs=()
v_feat_dirs+=(${feat_root}/video_features)

# # text features
t_feat_dir=${feat_root}/query_features/ # maybe not used
t_feat_dim=512

#### training
bsz=4
lr=1e-3
enc_layers=3
dec_layers=3
t2v_layers=2
moment_layers=1
sent_layers=1
num_prompts=1

# BK BT DS FM GA MS PK PR VT VU
# BK: bsz=4 seed=2018 dropout=0.2 n_epoch=1000 0.94299
# BT: bsz=4 seed=2018 dropout=0.1 n_epoch=1000 0.89021  no-sota
# DS: bsz=4 seed=2018 dropout=0.1 n_epoch=1000 0.79472
# FM: bsz=4 seed=2018 dropout=0.1 n_epoch=1000 0.74365  no-sota
# GA: bsz=4 seed=42 dropout=0.1  n_epoch=1000 0.9149  no-sota
# MS: bsz=4 seed=3407 dropout=0.2  n_epoch=1000 0.89194  
# PK: bsz=4 seed=3407 dropout=0.2  n_epoch=1000  0.87458
# PR: bsz=4 seed=3407 dropout=0.1  n_epoch=1000  0.89563
# VT: bsz=4 seed=2018 dropout=0.1  n_epoch=1000  0.87833 no-sota
# VU: bsz=4 seed=2018 dropout=0.2  n_epoch=1000  0.94104
######## TVSUM domain name
for dset_domain in MS
do
    for seed in 2018
    do
        PYTHONPATH=$PYTHONPATH:. python upm/train.py \
        --dset_name ${dset_name} \
        --ctx_mode ${ctx_mode} \
        --train_path ${train_path} \
        --eval_path ${eval_path} \
        --eval_split_name ${eval_split_name} \
        --v_feat_dirs ${v_feat_dirs[@]} \
        --v_feat_dim ${v_feat_dim} \
        --t_feat_dir ${t_feat_dir} \
        --t_feat_dim ${t_feat_dim} \
        --bsz ${bsz} \
        --results_root ${results_root}/${dset_domain} \
        --exp_id ${exp_id} \
        --max_v_l 1000 \
        --n_epoch 1000 \
        --lr_drop 2000 \
        --max_es_cnt -1 \
        --seed $seed \
        --lr ${lr} \
        --dset_domain ${dset_domain} \
        --enc_layers ${enc_layers} \
        --dec_layers ${dec_layers} \
        --t2v_layers ${t2v_layers} \
        --moment_layers ${moment_layers} \
        --sent_layers ${sent_layers} \
        --num_prompts ${num_prompts} \
        --total_prompts 10 \
        ${@:1}
    done
done
