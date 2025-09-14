from typing import List
from api.utils.minio_client import get_minio_client

import logging
logger = logging.getLogger(__name__)

def load_image_bytes_from_doc_ids(doc_ids: List[str]) -> List[bytes]:
    # 惰性导入，避免 REPL 导入期拉起整个服务栈
    from api.db.services.document_service import DocumentService
    from api.db.services.file_service import File2DocumentService  # ← 关键

    mc = get_minio_client()
    out: List[bytes] = []
    for doc_id in (doc_ids or []):
        # 1) 先确认 doc 存在（可选，方便日志）
        exist, doc = DocumentService.get_by_id(doc_id)
        if not exist or not doc:
            logger.warning("blob_loader: document not found: %s", doc_id)
            continue
        try:
            # 2) 用官方服务层拿“桶对象键”，不要自己拼
            bucket, object_name = File2DocumentService.get_storage_address(doc_id=doc_id)
        except Exception as e:
            logger.warning("blob_loader: get_storage_address failed for %s: %s", doc_id, e, exc_info=True)
            continue
        try:
            resp = mc.get_object(bucket, object_name)
            data = resp.read()
            if data:
                logger.info("blob_loader: loaded %s/%s bytes=%d", bucket, object_name, len(data))
                out.append(data)
            else:
                logger.warning("blob_loader: empty object for %s/%s", bucket, object_name)
        except Exception as e:
            logger.warning("blob_loader: failed to read %s/%s: %s", bucket, object_name, e, exc_info=True)
            continue
    return out