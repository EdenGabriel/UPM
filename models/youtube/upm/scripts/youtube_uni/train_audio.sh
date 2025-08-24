dset_name=youtube_uni
ctx_mode=video_tef
v_feat_types=vgg
t_feat_type=clip 
a_feat_type=pann
results_root=results_audio_youtubeuni
exp_id=exp


######## data paths
# train_path=data/youtube_uni/youtube_train.jsonl
# eval_path=data/youtube_uni/youtube_anno.jsonl
train_path=data/youtube_uni/youtube_train.jsonl
eval_path=data/youtube_uni/youtube_valid.jsonl
eval_split_name=val

######## setup video+text features
# feat_root=../features/tvsum
feat_root=/home/data/yj/DATA/youtube_uni

# # video features
# v_feat_dim=2816
# v_feat_dirs=()
# v_feat_dirs+=(${feat_root}/vid_clip)
# v_feat_dirs+=(${feat_root}/vid_slowfast)
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"vgg"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_rgb_opt_features)
  (( v_feat_dim += 2048 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi

# # text features
t_feat_dir=${feat_root}/txt_clip/ # maybe not used
t_feat_dim=300


# audio features
if [[ ${a_feat_type} == "pann" ]]; then
  a_feat_dir=${feat_root}/audio_features/
  a_feat_dim=2050
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=4
lr=2e-4
enc_layers=3
dec_layers=3
t2v_layers=2
moment_layers=1
sent_layers=1
n_epoch=2000
total_prompts=10
# Dog Gym. Par. Ska. Ski. Sur.
# dog gymnastics parkour skating skiing surfing
# seed 2018 42 3407
for seed in 2020
do 
    for dset_domain in skiing 
    do
        for dropout in 0.1
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
            --a_feat_dir ${a_feat_dir} \
            --a_feat_dim ${a_feat_dim} \
            --bsz ${bsz} \
            --results_root ${results_root}/${dset_domain}_${seed}_${dropout}_${lr}_${n_epoch}_${bsz}_muv30t5a30_seed \
            --total_prompts ${total_prompts}\
            --exp_id ${exp_id} \
            --max_v_l 1000 \
            --n_epoch ${n_epoch} \
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
            --clip_length 1 \
            --lw_saliency 1 \
            --total_prompts 10 \
            --num_workers 0
            ${@:1}
        done
    done
done

