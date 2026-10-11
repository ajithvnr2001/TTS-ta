import sys
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout.reconfigure(encoding='utf-8')
api = KaggleApi()
api.authenticate()

res = api.kernels_logs('ajithkumaevnr2001/indic-parler-tts')
with open('kernel_v2_error.log', 'w', encoding='utf-8') as f:
    if isinstance(res, list):
        for entry in res:
            f.write(f"[{entry.get('stream_name')}] {entry.get('data')}\n")
    elif isinstance(res, dict):
        f.write(str(res))
    else:
        f.write(str(res))

print("Successfully written kernel_v2_error.log!")
