import subprocess
import sys
import time

from audio_censor import gpu


def test_second_run_waits_for_the_gpu(tmp_path, capsys):
    code = ("import time; from audio_censor import gpu\n"
            "with gpu.turn():\n    print('held', flush=True); time.sleep(1.5)\n")
    holder = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    t0 = time.time()
    with gpu.turn():
        waited = time.time() - t0
    holder.wait()
    assert waited > 0.5 and "waiting to start scanning" in capsys.readouterr().err
