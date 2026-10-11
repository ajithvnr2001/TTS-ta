import sys, io
import pandas as pd
from huggingface_hub import hf_hub_download
import soundfile as sf

sys.stdout.reconfigure(encoding='utf-8')
p = hf_hub_download('SPRINGLab/IndicVoices-R_Tamil', 'data/train-00000-of-00014.parquet', repo_type='dataset')
print('Downloaded to:', p)
df = pd.read_parquet(p)
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)

for i in range(5):
    row = df.iloc[i]
    audio_data = row.get('audio', {})
    audio_bytes = audio_data.get('bytes', b'')
    sr = audio_data.get('sampling_rate')
    text = row.get('text', row.get('transcription', ''))
    
    # Read audio duration
    if audio_bytes:
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        dur = len(data) / samplerate
        print(f"Sample {i}: duration={dur:.2f}s, sr={samplerate}Hz, text={text}")
        if i == 0:
            # Save the first sample locally as reference
            out_sample = r'D:\azure\indic-parler-tts-kaggle\indicvoices_sample_tamil.wav'
            sf.write(out_sample, data, samplerate)
            print(f"Saved sample 0 to: {out_sample}")
