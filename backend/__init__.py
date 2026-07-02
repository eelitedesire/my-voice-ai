"""Production-grade speaker diarization + recognition backend (SpeechBrain + Sherpa).

The KMP_DUPLICATE_LIB_OK guard protects against a duplicate OpenMP runtime when
torch (SpeechBrain) and onnxruntime (Sherpa-ONNX ASR) coexist in one process.
It must be set before torch/onnxruntime are imported, so it lives here.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
