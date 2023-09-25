# -*- coding: utf-8 -*-

from __future__ import absolute_import

import tarfile
from io import BytesIO
from typing import Dict



def compress_gz(data: Dict[str, str]) -> bytes:
    """
    以 gz 格式进行压缩


    """
    outbuf = BytesIO()

    with tarfile.open(fileobj=outbuf, mode="w:gz") as tar:
        for k, v in data.items():
            info = tarfile.TarInfo(name=k)
            v_bytes = v.encode("utf8")
            info.size = len(v_bytes)
            tar.addfile(info, fileobj=BytesIO(v_bytes))

    return outbuf.getvalue()


def decompress_gz(data: bytes) -> Dict[str, str]:
    ret = {}
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                ret[member.name] = tar.extractfile(member).read().decode("utf8")

    return ret


def compress(data: Dict[str, str], method: str = "gz") -> bytes:
    if method != "gz":
        raise NotImplementedError("Only 'gz' method is supported by now")
    return compress_gz(data)


def decompress(data: bytes, method: str = "gz") -> Dict[str, str]:
    if method != "gz":
        raise NotImplementedError("Only 'gz' method is supported by now")
    return decompress_gz(data)


def get_compress_file_ext(method: str) -> str:
    if method != "gz":
        raise NotImplementedError("Only 'gz' method is supported by now")
    return "tar.gz"
