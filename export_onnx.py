import torch 
from PIL import Image
import requests

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cpu"
model, preprocess = load_from_name("ViT-L-14-336", device=device, download_root='./')
model.eval()

# import image onnx
# print(preprocess)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)

torch.onnx.export(model,
            (image, None),
            "./cnclip_vit_l14_336px_image_encoder.onnx",
            input_names=['image'],
            output_names=['unnorm_image_features'],
            export_params=True,
            do_constant_folding=False,
            opset_version=14,)

# import text onnx
text = clip.tokenize(["a diagram"]).to("cpu")

torch.onnx.export(model,
            (None, text),
            "./cnclip_vit_l14_336px_bert_encoder.onnx",
            input_names=['text'],
            output_names=['unnorm_text_features'],
            opset_version=14,)
