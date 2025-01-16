import io
import tarfile

filename = "sample-1.0-0.tar.bz2"
content = {
    "info/index.json": b'{}',
    "info/paths.json": b'{"paths":[],"paths_version":1}',
}

with tarfile.open(filename, mode="w:bz2") as tf:
    for name, payload in content.items():
        ti = tarfile.TarInfo(name)
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))
