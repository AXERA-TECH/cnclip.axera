# CLIP.axera
[cnclip-vit-l-14](https://github.com/OFA-Sys/Chinese-CLIP) demo on axera

## 支持平台
- [x] AX650N
- [ ] AX630C

### requirements.txt

需要安装 cnclip , `pip install cn_clip`

### 导出模型(PyTorch -> ONNX)
```
python export_onnx.py
```
导出成功后会生成两个onnx模型:
- image encoder: cnclip_vit_l14_336px_image_encoder.onnx
- text encoder: cnclip_vit_l14_336px_bert_encoder.onnx


#### 转换模型(ONNX -> Axera)
使用模型转换工具 Pulsar2 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 .axmodel，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 Pulsar2 build 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考[AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)


#### 量化数据集准备
此处仅用作demo，建议使用实际参与训练的数据
- image数据:

    下载[dataset_v04.zip](https://github.com/user-attachments/files/20480889/dataset_v04.zip)或自行准备

- text数据:
    ```
    python gen_text_calibration_data.py
    cd dataset
    zip -r bert_cali.zip bert_cali/
    ```
最终得到两个数据集：

\- dataset/dataset_v04.zip

\- dataset/bert_cali.zip

#### 模型编译
修改配置文件
检查config.json 中 calibration_dataset 字段，将该字段配置的路径改为上一步准备的量化数据集存放路径

image encoder的混合精度配置可以参考get_op_name.py得到op_name后自行更改config优化

在编译环境中，执行pulsar2 build参考命令：
```
# image encoder
pulsar2 build --config build_config/cnclip_vit_l14_336px_vision_u16u8.json --input cnclip_vit_l14_336px_image_encoder.onnx --output_dir build_output/image_encoder --output_name cnclip_vit_l14_336px_image_encoder.axmodel

# text encoder
pulsar2 build --config build_config/cnclip_vit_l14_336px_text_u16.json --input cnclip_vit_l14_336px_bert_encoder.onnx --output_dir build_output/text_encoder --output_name cnclip_vit_l14_336px_bert_encoder.axmodel
```

（给出了u16和混合精度两种image encoder的config）


编译完成后得到两个axmodel模型：


\- cnclip_vit_l14_336px_image_encoder.axmodel

\- cnclip_vit_l14_336px_bert_encoder.axmodel


### Python API 运行
需基于[PyAXEngine](https://github.com/AXERA-TECH/pyaxengine)在AX650N上进行部署

demo基于原repo中的提取图文特征向量并计算相似度，将demo/和编译好的axmodel拷贝到开发板上后，运行demo_onboard.py文件

(demo_onnx.py为对应的onnx版本,比较可知二者结果相似)

输入图片：

![](demo/pokemon.jpeg)

输入文本：

["杰尼龟", "妙蛙种子", "皮卡丘", "小火龙"]

输入类别置信度：

[0.01921699 0.033048   0.9386759  0.00905911]

```shell
$ python demo.py
[INFO] Available providers:  ['AxEngineExecutionProvider']
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.10.1s
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 4.0 685bfee4
image [1, 3, 336, 336] float32
unnorm_image_features [1, 768] float32
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 4.0 685bfee4
text [1, 52] int32
unnorm_text_features [1, 768] float32
Label probs: [[0.01921699 0.033048   0.9386759  0.00905911]]
```


## 技术讨论

- Github issues
- QQ 群: 139953715