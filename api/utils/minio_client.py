from __future__ import annotations
from minio import Minio
from functools import lru_cache
from urllib.parse import urlparse

# 使用 utils 提供的配置加载能力，避免依赖不存在的 settings.SERVICE_CONF
from api.utils import file_utils
try:
    # conf_realpath 在多数版本存在；若不存在可退回手工拼路径
    from api.utils import conf_realpath  # type: ignore
except Exception:
    import os
    def conf_realpath(name: str) -> str:
        return os.path.join("/ragflow", "conf", name)

SERVICE_CONF_PATH = conf_realpath("service_conf.yaml")

@lru_cache(maxsize=1)
def _load_minio_conf() -> dict:
    """
    加载 /ragflow/conf/service_conf.yaml 并取出 minio 段。
    由 docker 的 entrypoint 根据 service_conf.yaml.template 渲染生成。
    """
    conf = file_utils.load_yaml_conf(SERVICE_CONF_PATH)
    if not isinstance(conf, dict):
        raise EnvironmentError(f"Invalid service_conf at: {SERVICE_CONF_PATH}")
    return conf.get("minio", {}) or {}

@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    mconf = _load_minio_conf()
    host = str(mconf.get("host", "minio:9000"))   # 允许是 http://minio:9000 也允许裸 host:port
    user = str(mconf.get("user", "rag_flow"))
    password = str(mconf.get("password", "infini_rag_flow"))

    # 兼容 host 既可能是 "minio:9000" 也可能是 "http://minio:9000"
    if host.startswith("http://") or host.startswith("https://"):
        u = urlparse(host)
        endpoint = u.netloc
        secure = (u.scheme == "https")
    else:
        endpoint = host
        secure = False

    return Minio(
        endpoint,
        access_key=user,
        secret_key=password,
        secure=secure,
    )

# from api.utils.blob_loader import load_image_bytes_from_doc_ids
# bs = load_image_bytes_from_doc_ids(["63a2b0967fd211f0b5fa8e9a7105b432"])
# print(len(bs), [len(b) for b in bs[:1]])