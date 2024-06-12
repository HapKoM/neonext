import os
os.environ["MOX_SILENT_MODE"] = "1"

import time
try:
    import moxing as mox
    mox.file.set_auth(is_secure=False)
except ImportError:
    mox = None


def in_cloud():
    return mox is not None


def mox_copy(src, dst, parallel=None):
    if src == dst:
        print("mox_copy: src=dst={}, return".format(src))
        return
    if not (src.startswith("s3://") or dst.startswith("s3://")):
        print(
            "mox_copy: at least one of src and dst need startswith s3://, "
            "src={}, dst={}, return".format(src, dst)
        )
        return

    if parallel is None:
        if src.startswith("s3://"):
            parallel = mox.file.is_directory(src)
        else:
            parallel = os.path.isdir(src)

    failed = 0
    while True:
        try:
            if parallel:
                mox.file.copy_parallel(src, dst)
            else:
                mox.file.copy(src, dst)
            break
        except Exception as e:
            failed += 1
            time.sleep(60)
            if failed % 10 == 0:
                print("error, maybe need check. copy failed {} times from {} to {}".format(failed, src, dst))
                print("error message: {}".format(e))
