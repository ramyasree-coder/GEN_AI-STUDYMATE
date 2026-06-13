import os
import subprocess
import time
import urllib.request


def is_up():
    try:
        with urllib.request.urlopen("http://localhost:8501", timeout=5) as r:
            print("STATUS", r.status)
            return True
    except Exception as e:
        print("NOT_UP", e)
        return False


if __name__ == '__main__':
    if is_up():
        print("ALREADY_UP")
        raise SystemExit(0)
    venv_streamlit = os.path.join(os.path.dirname(__file__), '..', '.venv', 'Scripts', 'streamlit.exe')
    venv_streamlit = os.path.normpath(venv_streamlit)
    if not os.path.exists(venv_streamlit):
        # try top-level .venv
        venv_streamlit = os.path.join('.', '.venv', 'Scripts', 'streamlit.exe')
    if not os.path.exists(venv_streamlit):
        print('NO_STREAMLIT_EXE')
        raise SystemExit(2)
    print('STARTING', venv_streamlit)
    p = subprocess.Popen([venv_streamlit, 'run', 'app.py'], cwd=os.path.abspath('.'))
    print('PID', p.pid)
    # wait a bit for server to start
    for i in range(20):
        time.sleep(1)
        if is_up():
            print('STARTED')
            raise SystemExit(0)
    print('FAILED_TO_START')
    raise SystemExit(3)
