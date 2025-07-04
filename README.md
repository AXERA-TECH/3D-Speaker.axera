# 3D-Speaker.
3D-Speaker DEMO on Axera

- 目前支持  Python 语言 
- 预编译模型下载[models](https://github.com/wzf19947/PPOCR_v5/releases/download/v1.0.0/model.tar.gz)。如需自行转换请参考[模型转换](/model_convert/README.md)

## 支持平台

- [x] AX650N
- [ ] AX630C

## 模型转换

[模型转换](./model_convert/README.md)

## 上板部署

- AX650N 的设备已预装 Ubuntu22.04
- 以 root 权限登陆 AX650N 的板卡设备
- 链接互联网，确保 AX650N 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备：AX650N DEMO Board

### Python API 运行

#### Requirements

```
cd python
pip3 install -r requirements.txt
``` 

#### 运行

##### 基于 ONNX Runtime 运行  
可在开发板或PC运行 

在开发板或PC上，运行以下命令  
```  
cd python
python3 run_onnx.py
```
输出结果
```
[INFO]: Computing the similarity score...
[INFO]: The similarity score between two input wavs is 0.7295
```

##### 基于AXEngine运行  
在开发板上运行命令

输入相同人音频数据
```
cd python  
python3 run_axmodel.py --wav ./wav/speaker1_a_cn_16k.wav ./wav/speaker1_b_cn_16k.wav
```  
输出结果：
```  
[INFO] Available providers:  ['AxEngineExecutionProvider']
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.12.0s
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 4.0 3352e83b
call __del__
call _unload
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 4.0 3352e83b
call __del__
call _unload
[INFO]: Computing the similarity score...
[INFO]: The similarity score between two input wavs is 0.7136
```  

输入不同人音频数据
```
python3 run_axmodel.py --wav ./wav/speaker1_a_cn_16k.wav ./wav/speaker2_a_cn_16k.wav
```
输出结果： 
```
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.12.0s
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 4.0 3352e83b
call __del__
call _unload
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 4.0 3352e83b
call __del__
call _unload
[INFO]: Computing the similarity score...
[INFO]: The similarity score between two input wavs is 0.0788
```

运行参数说明:  
| 参数名称 | 说明  |
| --- | --- | 
| --model | 检测模型路径 | 
| --wavs | 输入.wav格式音频数据 | 
| --samplerate | 音频采样率 | 
| --max_frames | 音频最大帧数 | 

### Latency

#### AX650N

| model | latency(ms) |
|---|---|
|ERes2NetV2|5.09|



## 技术讨论

- Github issues
- QQ 群: 139953715
