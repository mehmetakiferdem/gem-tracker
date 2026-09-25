import time, sys
_t = time.monotonic()
sys.argv = ["app_edgeai.py", "../configs/tracking.yaml", "-n"]
import prof_infer_pipe
print("[PROF] import prof %.2fs" % (time.monotonic() - _t), flush=True)
import app_edgeai
print("[PROF] import app %.2fs" % (time.monotonic() - _t), flush=True)
app_edgeai.main(sys.argv)
