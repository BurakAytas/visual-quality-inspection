| Format | Size (MB) | Median ms/image | p95 ms/image | Image AUROC | Recall | False-alarm rate |
|---|---|---|---|---|---|---|
| torch_fp32 | - | 315.7 | 438.5 | 1.000 | 1.000 | 0.125 |
| onnx_fp32 | 213 | 229.8 | 302.7 | 1.000 | 1.000 | 0.125 |
| openvino_fp32 | 213 | 195.5 | 251.2 | 1.000 | 1.000 | 0.125 |
| openvino_fp16 | 106 | 192.1 | 231.9 | 1.000 | 1.000 | 0.125 |
| openvino_int8 | 171 | 167.4 | 227.5 | 1.000 | 1.000 | 0.125 |
