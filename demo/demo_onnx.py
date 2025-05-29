from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2

# import axengine as axe
from bert_tokenizer import FullTokenizer

def tokenize(texts):
    context_length = 52
    _tokenizer = FullTokenizer()
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[
                                                        :context_length - 2] + [_tokenizer.vocab['[SEP]']])

    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = np.array(tokens)

    return result

def get_image_encoder():
    image_encoder = ort.InferenceSession("cnclip_vit_l14_336px_image_encoder.onnx")

    for input in image_encoder.get_inputs():
        print(input.name, input.shape, input.type)

    for output in image_encoder.get_outputs():
        print(output.name, output.shape, output.type)
    
    return image_encoder

def get_text_encoder():
    text_encoder = ort.InferenceSession("cnclip_vit_l14_336px_bert_encoder.onnx")

    for input in text_encoder.get_inputs():
        print(input.name, input.shape, input.type)

    for output in text_encoder.get_outputs():
        print(output.name, output.shape, output.type)
    
    return text_encoder

def get_similarity(image, text):
    image_features = image
    text_features = text

    # normalized features
    image_features = image_features/np.linalg.norm(image_features, ord=None, axis=-1, keepdims=True)
    text_features = text_features / np.linalg.norm(text_features, ord=None, axis=-1, keepdims=True)

    # cosine similarity as logits
    logit_scale = np.exp(np.log(100))
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text

def preprocess(image, width=336, height=336):
    # data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = np.array(image)
    data = cv2.resize(data, (width, height)).astype(np.float32)
    data = data.astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    data = (data - mean) / std
    data = data.transpose(2, 0, 1).reshape(1, 3, height, width)
    return data

def softmax(x, axis=-1):
    """
    对输入数组沿指定轴计算softmax
    
    参数:
    x (np.ndarray): 输入数组
    axis (int): 要计算softmax的轴，默认为最后一维
    
    返回:
    np.ndarray: softmax结果，形状与输入相同
    """
    # 为避免数值溢出，先减去最大值
    x_max = np.max(x, axis=axis, keepdims=True)
    x = x - x_max
    
    # 计算指数和归一化
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

if __name__ == "__main__":
    image_encoder = get_image_encoder()
    text_encoder = get_text_encoder()

    device = "cpu"
    # _, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')

    image = preprocess(Image.open("demo/pokemon.jpeg"))
    texts = tokenize(["杰尼龟", "妙蛙种子", "皮卡丘", "小火龙"])
    

    image_features = image_encoder.run(None, {"image": image})[0]
    text_features = []
    for t in texts:
        text_features.append(text_encoder.run(None,{"text": [t]}))
    image_features = np.array(image_features)
    text_features = np.array([t[0][0] for t in text_features])
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务

    logits_per_image, logits_per_text = get_similarity(image_features, text_features)
    
    probs = softmax(logits_per_image)

    print("Label probs:", probs)