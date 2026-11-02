| Format | Size (MB) | Median ms/image | p95 ms/image | Image AUROC | Recall | False-alarm rate |
|---|---|---|---|---|---|---|
| torch_fp32 | - | 286.7 | 301.6 | 1.000 | 1.000 | 0.125 |
| onnx_fp32 | 213 | 218.0 | 234.5 | 1.000 | 1.000 | 0.125 |
| openvino_fp32 | 213 | 177.3 | 197.3 | 1.000 | 1.000 | 0.125 |
| openvino_fp16 | 106 | 184.9 | 201.3 | 1.000 | 1.000 | 0.125 |
| openvino_int8 | 142 | 154.4 | 169.6 | 1.000 | 1.000 | 0.250 |
