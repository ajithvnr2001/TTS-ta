import urllib.request
import json
import os

os.makedirs('indicvoices_samples', exist_ok=True)
url = 'https://datasets-server.huggingface.co/rows?dataset=SPRINGLab/IndicVoices-R_Tamil&config=default&split=train&offset=0&limit=5'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        rows = data.get('rows', [])
        print(f"Total rows fetched: {len(rows)}")
        for idx, r in enumerate(rows):
            row = r.get('row', {})
            text = row.get('normalized') or row.get('text')
            spk = row.get('speaker_id')
            gender = row.get('gender')
            district = row.get('district')
            dur = row.get('duration')
            print(f"Sample {idx+1}: spk={spk}, gender={gender}, district={district}, dur={dur}s")
            
            audio_info = row.get('audio')
            audio_src = None
            if isinstance(audio_info, list) and len(audio_info) > 0:
                audio_src = audio_info[0].get('src')
            elif isinstance(audio_info, dict):
                audio_src = audio_info.get('src')
            
            if audio_src:
                out_path = os.path.join('indicvoices_samples', f'indicvoices_sample_{idx+1}_{gender}_{district}.wav')
                urllib.request.urlretrieve(audio_src, out_path)
                print(f"  -> Downloaded: {out_path} ({os.path.getsize(out_path)} bytes)")
                txt_path = os.path.join('indicvoices_samples', f'indicvoices_sample_{idx+1}_{gender}_{district}.txt')
                with open(txt_path, 'w', encoding='utf-8') as tf:
                    tf.write(f"Speaker: {spk}\nGender: {gender}\nDistrict: {district}\nDuration: {dur}\nText: {text}\n")
except Exception as e:
    print(f"Error: {e}")
