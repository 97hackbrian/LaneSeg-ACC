# LaneSeg-ACC

steps:
/usr/src/tensorrt/bin/trtexec \
  --onnx=segformer_fixed_unique.onnx \
  --saveEngine=cityscapes_fan_tiny_hybrid_224.plan \
  --fp16 \
  --minShapes=unique_tensor_0:1x3x224x224 \
  --optShapes=unique_tensor_0:1x3x224x224 \
  --maxShapes=unique_tensor_0:1x3x224x224 \
  --tacticSources=-CUBLAS,-CUBLAS_LT,-CUDNN,+EDGE_MASK_CONVOLUTIONS \
  --verbose

  
