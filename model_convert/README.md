# 模型转换

## 创建虚拟环境

```
git clone https://github.com/alibaba-damo-academy/3D-Speaker.git && cd 3D-Speaker
conda create -n 3D-Speaker python=3.8
conda activate 3D-Speaker
pip install -r requirements.txt
```

## 导出模型（ONNX）
导出onnx可以参考：
```
https://github.com/modelscope/3D-Speaker/blob/main/runtime/onnxruntime/README.md
```

一般情况下，用以下命令即可将ERes2NetV2模型导出为onnx模型：
```
python speakerlab/bin/export_speaker_embedding_onnx.py \
    --experiment_path your/experiment_path/ \
    --model_id iic/speech_eres2netv2_sv_zh-cn_16k-common \ # you can use other model_id
    --target_onnx_file path/to/save/onnx_model
```
但直接导出的onnx在使用AX工具生成.axmodel存在一些问题，为了适配AX平台，导出onnx模型需要做的修改

如下：

1、增加输入层ch维度
修改文件：
```
https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/models/eres2net/ERes2NetV2.py
```
修改如下：
```
class ERes2NetV2(nn.Module):
    def forward(self, x):
        #x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        #x = x.unsqueeze_(1)
        x = x.permute(0, 1, 3, 2)  # (B,T,F) => (B,F,T)
```

2、固定输入层shape大小
```
-The model input shape is (batch_size, frame_num, feature_dim).
```
-输入层为fbank特征：feature_dim:80，frame_num则与输入音频长度相关，本例中设为固定值:360，对应音频时长（16K）约为3s左右
修改文件：
```
https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/bin/export_speaker_embedding_onnx.py
```
修改export_onnx_file接口如下：
```
def export_onnx_file(model, target_onnx_file):
    dummy_input = torch.randn(1, 1, 360, 80)
    torch.onnx.export(model,
                        dummy_input,
                        target_onnx_file,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['feature'],
                        output_names=['embedding'],
                        dynamic_axes={})
    logger.info(f"Export model onnx to {target_onnx_file} finished")
```

导出成功后会生成'res2netv2.onnx'模型.

## 动态onnx转静态
```
onnxsim res2netv2.onnx  res2netv2_sim_static.onnx --overwrite-input-shape=1,1,360,80
```

## 转换模型（ONNX -> Axera）
使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

### 下载量化数据集
```
bash download_dataset.sh
```
这个模型的输入是音频fank特征，需要将生成fbank特征保存为.npy文件，再进行量化 

### 模型转换

#### 修改配置文件
 
检查`config.json` 中 `calibration_dataset` 字段，将该字段配置的路径改为上一步下载的量化数据集存放路径  

#### Pulsar2 build

参考命令如下：

```
pulsar2 build --input res2netv2.onnx --config ./res2netv2.json --output_dir ./ourput --output_name res2netv2.axmodel  --target_hardware AX650 --compiler.check 0

也可将参数写进json中，直接执行：
pulsar2 build --config ./res2netv2.json
```
