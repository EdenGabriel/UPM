'''
import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

# load video
video_path = "/home/yj/CGDETR-v/run_on_video/example/RoripwjYFp8_60.0_210.0.mp4"
container = av.open(video_path)

# extract evenly spaced frames from video
seg_len = 75
clip_len = model.config.encoder.num_frames
print(clip_len,seg_len)
indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
frames = []
container.seek(0)
for i, frame in enumerate(container.decode(video=0)):
    if i in indices:
        frames.append(frame.to_ndarray(format="rgb24"))

# generate caption
gen_kwargs = {
    "min_length": 10, 
    "max_length": 20, 
    "num_beams": 8,
}
pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
print(pixel_values.shape)
pixel_values = torch.randn(1,8,256)
tokens = model.generate(pixel_values, **gen_kwargs)
caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
print(caption) # A man and a woman are dancing on a stage in front of a mirror.
'''
import numpy as np
# 加载 .npz 文件
vid_data = np.load('/home/yj/CGDETR-v/multimodal_reps/reps/_0u5I0OJP6U_60.0_210.0.npz')
mask_data = np.load('/home/yj/CGDETR-v/multimodal_reps/rep_masks/_0u5I0OJP6U_60.0_210.0.npz')
# print(vid_data['data'].shape)
print(mask_data['data'])