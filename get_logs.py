import sys
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout.reconfigure(encoding='utf-8')
api = KaggleApi()
api.authenticate()

res = api.kernels_logs('ajithkumaevnr2001/indic-parler-tts')
with open('kernel_v5_error.log', 'w', encoding='utf-8') as f:
    if isinstance(res, list):
        for entry in res:
            sn = entry.get('stream_name')
            d = entry.get('data')
            f.write(f"[{sn}] {d}")
    else:
        f.write(str(res))

print("Successfully written kernel_v5_error.log!")
with open('kernel_v5_error.log', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print("".join(lines[-40:]))
