import os
import hashlib
from pathlib import Path
from typing import Literal, Generator

import requests
import numpy as np
import onnxruntime as ort

from candies.logging import log
from candies.base import Candies, CandiesError

REGISTRY = {
    "a": {
        "url": "https://zenodo.org/api/records/15699208/files/model_a.onnx/content",
        "md5": "7a8a627129817418963c7b77b962e0bd",
        "size_mb": 114.80,
    },
    "b": {
        "url": "https://zenodo.org/api/records/15699208/files/model_b.onnx/content",
        "md5": "aecb4c022e673f80f6cf73ced9e4c373",
        "size_mb": 87.28,
    },
    "c": {
        "url": "https://zenodo.org/api/records/15699208/files/model_c.onnx/content",
        "md5": "027ab7bf064944b9782ab51ef1dd8416",
        "size_mb": 139.52,
    },
    "d": {
        "url": "https://zenodo.org/api/records/15699208/files/model_d.onnx/content",
        "md5": "9d33f3c9e3db15b5903ada9043e93126",
        "size_mb": 157.33,
    },
    "e": {
        "url": "https://zenodo.org/api/records/15699208/files/model_e.onnx/content",
        "md5": "d40fff195b94c75d0842660b913a3bc5",
        "size_mb": 164.86,
    },
    "f": {
        "url": "https://zenodo.org/api/records/15699208/files/model_f.onnx/content",
        "md5": "e5391f7b501f2b50666438015b2221f1",
        "size_mb": 114.00,
    },
    "g": {
        "url": "https://zenodo.org/api/records/15699208/files/model_g.onnx/content",
        "md5": "1282ebaff6999e93d091dc0a689131af",
        "size_mb": 139.52,
    },
    "h": {
        "url": "https://zenodo.org/api/records/15699208/files/model_h.onnx/content",
        "md5": "60c5d33688573a7498190238840c8752",
        "size_mb": 114.80,
    },
    "i": {
        "url": "https://zenodo.org/api/records/15699208/files/model_i.onnx/content",
        "md5": "5c21e392e2517bdede04034f188876d3",
        "size_mb": 132.58,
    },
    "j": {
        "url": "https://zenodo.org/api/records/15699208/files/model_j.onnx/content",
        "md5": "7beac5476ff03f6fa96ec304bf44acb1",
        "size_mb": 301.24,
    },
    "k": {
        "url": "https://zenodo.org/api/records/15699208/files/model_k.onnx/content",
        "md5": "b8e0c9a24275a1813bd007088e1b19f7",
        "size_mb": 116.16,
    },
}


def onnxdir() -> Path:
    return (
        Path(onnxhome)
        if (onnxhome := os.environ.get("ONNX_HOME"))
        else Path(os.environ.get("HOME", os.getcwd())) / "onxxmodels"
    )


def calcmd5(fn: str | Path) -> str:
    md5 = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def loadmodel(ix: str) -> Path:
    if ix not in REGISTRY:
        raise CandiesError(f"MODEL {ix} NOT FOUND IN REGISTRY. ABORT.")
    modeldir = onnxdir()
    modelinfo = REGISTRY[ix]
    modelpath = modeldir / f"model_{ix}.onnx"
    if modelpath.exists():
        if calcmd5(modelpath) == modelinfo["md5"]:
            return modelpath
        else:
            modelpath.unlink()
    try:
        response = requests.get(modelinfo["url"], stream=True)
        response.raise_for_status()

        downloaded = 0
        totalsize = int(response.headers.get("content-length", 0))
        with open(modelpath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if totalsize > 0:
                        progress = (downloaded / totalsize) * 100
                        print(f"\rPROGRESS: {progress:.1f}%", end="", flush=True)
        print()

        if calcmd5(modelpath) != modelinfo["md5"]:
            modelpath.unlink()
            raise CandiesError(f"FAILED TO VERIFY HASH FOR MODEL {ix}. ABORT.")
        return modelpath
    except Exception as e:
        if modelpath.exists():
            modelpath.unlink()
        raise CandiesError(f"FAILED TO DOWNLOAD MODEL {ix}: {str(e)}. ABORT.")


def batchify(candies: Candies, batchsize: int = 8) -> Generator:
    for i in range(0, len(candies), batchsize):
        batch, ddbatch, dmtbatch = [], [], []
        for candy in candies[i : i + batchsize]:
            batch.append(candy)
            dd = candy.dedispersed.data
            dmt = candy.dmtransform.data
            ddbatch.append(dd.reshape(*dd.shape, 1))
            dmtbatch.append(dmt.reshape(*dmt.shape, 1))
        ddbatch = np.array(ddbatch)
        dmtbatch = np.array(dmtbatch)
        yield batch, ddbatch, dmtbatch


def classify(
    candies: Candies,
    gpuid: int = -1,
    batchsize: int = 8,
    modelid: Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"] = "a",
) -> Candies:
    providers = []
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers.append(("CUDAExecutionProvider", {"device_id": f"{gpuid}"}))
    elif "ROCMExecutionProvider" in available_providers:
        providers.append(("ROCMExecutionProvider", {"device_id": f"{gpuid}"}))
    providers.append("CPUExecutionProvider")

    modeldir = onnxdir()
    modeldir.mkdir(exist_ok=True)
    modelpath = loadmodel(modelid)
    onnxsession = ort.InferenceSession(str(modelpath), providers=providers)

    labeled = []
    for batch, ddbatch, dmtbatch in batchify(candies=candies, batchsize=batchsize):
        inputs = {}
        names = [_.name for _ in onnxsession.get_inputs()]
        for ix, name in enumerate(names):
            cleanname = name.split(":")[0]
            if ix == 0:
                inputs[cleanname] = ddbatch
            elif ix == 1:
                inputs[cleanname] = dmtbatch
            else:
                raise CandiesError(f"UNEXPECTED INPUT {ix}: {name}. ABORT.")
        outputs = onnxsession.run(None, inputs)
        for ix, probability in enumerate(np.asarray(outputs[0])[:, 1]):
            candy = batch[ix]
            candy.probability = probability
            candy.label = probability >= 0.5
            batch[ix] = candy
            log.info(f"Classification succeeded for {candy.id}.")
        labeled.extend(batch)
    return Candies(labeled)


__all__ = ["classify"]
