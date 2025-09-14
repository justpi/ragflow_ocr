# api/utils/ocr_utils.py
from typing import List, Optional
import io, base64, re, os
from PIL import Image
import numpy as np

import logging
logger = logging.getLogger(__name__)

# --- 环境变量&资源路径 ---
_DEEPDOC_DIR = os.getenv("DEEPDOC_RES_DIR", "/ragflow/rag/res/deepdoc")  # 与官方建议一致
_LANG = os.getenv("CHAT_IMAGE_OCR_LANG", "ch")  # 'ch' | 'en' | 'chinese_cht' 等

# --- PaddleOCR 可选 ---
_PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except Exception:
    _PADDLE_AVAILABLE = False

# --- DeepDoc OCR（整合类） ---
_DEEPDOC_AVAILABLE = False
_OCR_SINGLETON = None
try:
    # DeepDoc 在 RAGFlow 仓库下：deepdoc/vision/ocr.py
    # 其 OCR 类内部组合 TextDetector/TextRecognizer，并暴露 recognize / recognize_batch
    from deepdoc.vision.ocr import OCR  # 注意：不是 TextRecognizer/TextDetector 直接用
    _DEEPDOC_AVAILABLE = True
except Exception:
    _DEEPDOC_AVAILABLE = False

_dataurl_pat = re.compile(r"^data:image/[^;];base64,(?P<b64>.)$", re.I)

def _to_image_bytes(obj) -> Optional[bytes]:
    if obj is None:
        return None
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, str):
        m = _dataurl_pat.match(obj)
        if m:
            return base64.b64decode(m.group("b64"))
        try:
            return base64.b64decode(obj)
        except Exception:
            return None
    if isinstance(obj, dict):
        if "content" in obj:
            return _to_image_bytes(obj["content"])
    return None

def _get_deepdoc_ocr():
    """
    懒加载 DeepDoc OCR 单例。
    需要确保 _DEEPDOC_DIR 下 det.onnx / rec.onnx 等资源存在（参考官方 FAQ）。
    """
    global _OCR_SINGLETON
    if _OCR_SINGLETON is not None:
        return _OCR_SINGLETON
    if not _DEEPDOC_AVAILABLE:
        return None
    # 根据 DeepDoc 版本，OCR(...) 可能需要传资源目录/设备等参数，常见参数示例：
    #   OCR(model_dir=_DEEPDOC_DIR, device="cuda" or "cpu")
    # 如果不支持 device 参数，可省略；若部署无 GPU，可能强制走 CPU（见社区讨论）。
    try:
        _OCR_SINGLETON = OCR()  # 视版本签名调整
        return _OCR_SINGLETON
    except Exception:
        return logger.warning(f"Failed to use Deepdoc OCR API.")

def _paddle_ocr():
    if not _PADDLE_AVAILABLE:
        return None
    # try:
    return PaddleOCR(use_angle_cls=True, lang=_LANG, device="cpu", enable_mkldnn=True)
    # except Exception:
    #     return logger.warning(f"")

def ocr_images_to_text(images: List[object], max_chars: int = 4000) -> str:
    """
    统一入口：优先 DeepDoc，失败回退 Paddle。
    images: bytes / base64 / dataURL / {'mime','content'}
    """
    logger.info(f"images: {len(images)}.")
    if not images:
        return ""
    # 规范化为 PIL.Image 列表
    pil_list = []
    for it in images:
        b = _to_image_bytes(it)
        if not b:
            continue
        try:
            pil_list.append(Image.open(io.BytesIO(b)).convert("RGB"))
        except Exception:
            continue
    if not pil_list:
        return ""

    # ---- 路径 A：DeepDoc OCR（优先）----
    ocr_dd = _get_deepdoc_ocr()
    if ocr_dd is not None:
        # try:
        texts = []
        for idx, pil in enumerate(pil_list, 1):
            # __call__ 只接受单张 ndarray
            dd_out = ocr_dd(np.array(pil))
            # dd_out 是 [(box, (text, score)), ...]
            lines = []
            for item in (dd_out or []):
                try:
                    txt, score = item[1]
                except Exception:
                    txt, score = "", 0.0
                if txt:
                    lines.append(txt)
            block = "\n".join(lines).strip()
            if block:
                texts.append(f"[Image #{idx} OCR]\n{block}")
        out = "\n\n".join(texts).strip()
        if max_chars and len(out) > max_chars:
            out = out[:max_chars] + "\n[TRUNCATED]"
        if out:
            return out
        # except Exception as e:
        #     import logging
        #     logging.getLogger(__name__).warning(
        #         "DeepDoc OCR failed, fallback to Paddle: %s", e, exc_info=True
        #     )
        # except Exception:
        #     # 回退到 Paddle
        #     logger.warning(f"DeepOcr failed, fallback to Paddle OCR.")

    # ---- 路径 B：PaddleOCR 回退 ----
    # paddle = _paddle_ocr()
    # if paddle is not None:
    #     import numpy as np
    #     texts = []
    #     for idx, img in enumerate(pil_list, 1):
    #         arr = np.array(img)[:, :, ::-1]  # BGR
    #         res = paddle.ocr(arr) or []
    #         logger.info(f"Paddle OCR res: {res}.\n")
    #         lines = []
    #         for page in res:
    #             for line in page:
    #                 lines.append(line[1][0])
    #         block = "\n".join(lines).strip()
    #         if block:
    #             texts.append(f"[Image #{idx} OCR]\n{block}")
    #     out = "\n\n".join(texts).strip()
    #     if max_chars and len(out) > max_chars:
    #         out = out[:max_chars] + "\n[TRUNCATED]"
    #     logger.info(f"Paddle OCR: {out}.\n")
    #     return out

    # ---- 双双不可用时 ----
    return "[OCR unavailable – DeepDoc & PaddleOCR not available.]"
