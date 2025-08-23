[***] Learning Unified Patterns of Multimodalities for Video Temporal Grounding

**UPM**
===

<!-- [![Static Badge](https://img.shields.io/badge/arxiv-2404.09263-red)](https://arxiv.org/abs/2404.09263) -->
<!--[![Static Badge](https://img.shields.io/badge/LICENSE-blue)](https://github.com/EdenGabriel/TaskWeave/blob/master/LICENSE)-->
![GitHub Repo stars](https://img.shields.io/github/stars/EdenGabriel/UPM)
![GitHub forks](https://img.shields.io/github/forks/EdenGabriel/UPM)

### News
- [2025/08] We release the code about UPM.
<!-- - [2024/02/27] Our paper is accepted by CVPR2024.-->

### ✨Introduction
A fundamental challenge within the multimodal learning field lies in the heterogeneity of data across modalities (video, text, and audio). It leads to semantic gaps and cognitive offset. However, most existing methods have not yet effectively address this challenge. Inspired by the multi-sensory system of the brain, we introduce a novel architecture for Video Temporal Grounding (VTG) that learns the Unified Pattern of Multimodalities (UPM). It effectively captures representations with the unified pattern across diverse modalities to enhance semantic understanding. We utilize the proposed modality co-occurrence engine to capture unified pattern representations for diverse modalities. Different from the commonly used cross-attention, we propose an efficient inter-modality interaction mechanism with lower computational cost and fewer parameters to improve multimodal interaction efficiency. Moreover, we develop a novel consciousness caption experiment inspired by human intelligence to enrich the evaluation standard for multimodal alignment. Unlike prior most models, our model integrates common information carriers in the real world (video, text, and audio) and achieves impressive results on five datasets for different downstream tasks.

### 🔎Data Preparation/Installation/More Details
Please refer to [MomentDETR](https://github.com/jayleicn/moment_detr) for more details.

Please refer to [UMT](https://github.com/TencentARC/UMT) for more details.

Please refer to [QD-DETR](https://github.com/wjun0830/QD-DETR) for more details.

### 🔧Training and Evaluation
- Train(Take `QVHighlights` as an example)
```python 
bash upm/scripts/train.sh 
bash upm/scripts/train_audio.sh 
```
- Evaluation (Take `QVHighlights` as an example)
```python
bash upm/scripts/inference.sh results/{direc}/model_best.ckpt 'val'
bash upm/scripts/inference.sh results/{direc}/model_best.ckpt 'test'
```
- Consciousness Caption experiments:
please refer to `UPM/clip4caption/test.py`, more details in this file.
```python
python -m torch.distributed.launch test.py --init_model ./clip4caption_vit-b-32_model.bin --bert_model bert-base-uncased --do_lower_case --output_dir ./results --do_eval --d_model 512 --video_dim 512 --max_words 48 --batch_size_val 128 --num_thread_reader 16 --visual_num_hidden_layers 2 --n_display=50 --decoder_num_hidden_layers 3
```
### References
If you are using our code, please consider citing the following paper.

#### The implementation of this code is built upon [CGDETR](https://github.com/wjun0830/CGDETR) and [MomentDETR](https://github.com/jayleicn/moment_detr) and [QD-DETR](https://github.com/wjun0830/QD-DETR), and we would like to express gratitude for the open-source contribution of [MomentDETR](https://github.com/jayleicn/moment_detr), [QD-DETR](https://github.com/wjun0830/QD-DETR) and [UMT](https://github.com/TencentARC/UMT).
