"""
AI4Bharat Indic Parler-TTS: Part 2 (Beats 21 to 30)
Acts VI to X: Earth 2550, The Citadel of Chronos, Infiltration, The Master Architect, Overload & The Peaceful Dawn
Multi-GPU Concurrent Execution (cuda:0 & cuda:1)
"""

import os
import sys
import time
import json
import gc
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

print("=" * 85, flush=True)
print(" AI4BHARAT INDIC PARLER-TTS: CINEMATIC THRILLER PART 2 (BEATS 21 TO 30)")
print(" Narrator: Jaya (Native Tamil Narrative Cadence)")
print(" Audio Format: 44,100 Hz, 16-bit PCM Mono, Peak-Normalized")
print("=" * 85, flush=True)

def ensure_dependencies():
    print("[*] Installing runtime dependencies...", flush=True)
    pkgs = [
        "protobuf>=5.26.1",
        "soundfile",
        "accelerate",
        "transformers>=4.44.0"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir"] + pkgs, check=True)

    print("[*] Installing Parler-TTS...", flush=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "git+https://github.com/huggingface/parler-tts.git"
    ], check=True)
    print("[+] Dependencies installed!", flush=True)

ensure_dependencies()

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

num_gpus = torch.cuda.device_count()
print(f"[*] Detected {num_gpus} CUDA Device(s):", flush=True)
devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else ["cpu"]

output_dir = Path("/kaggle/working/output_part2") if Path("/kaggle/working").exists() else Path("./output_part2")
output_dir.mkdir(parents=True, exist_ok=True)

BEATS_PART2 = [
    {
        "beat": 21,
        "name": "beat_21_temporal_slipstream",
        "title": "Act VI: The Temporal Slipstream (கால வெளிப்பாதை)",
        "text": "எஸ்கேப் பாட் ஒளியின் வேகத்தைத் தாண்டி காலத்தின் சுரங்கத்திற்குள் பாய்ந்தது. சுற்றிலும் கோடிக்கணக்கான விண்மீன்கள் மின்னல் கோடுகளாக நீண்டு தெரிந்தன. கதிரின் நரம்புகளில் அந்த நீல கிரிஸ்டலின் தரவுகள் தீப்பிழம்பாகப் பாய்ந்தன. கணினி எச்சரித்தது: எச்சரிக்கை! காலச் சுழல் அதிவேகமாக சுழல்கிறது. இரண்டாயிரத்து ஐந்நூற்று ஐம்பதாம் ஆண்டை நெருங்குகிறோம்! கதிர் தன் கவச உடையை சரிசெய்து காக்பிட் கட்டுப்பாடுகளை தன் கைகளில் எடுத்தான்.",
        "desc": "Jaya speaks in a calm, clear, and atmospheric narrative tone with natural Tamil cadence and subtle mystery in a close-sounding environment. Very high quality audio with no background noise."
    },
    {
        "beat": 22,
        "name": "beat_22_arrival_earth_2550",
        "title": "Act VI: Arrival at Earth 2550 (எதிர்கால பூமி: சாம்பல் உலகம்)",
        "text": "காலச் சுரங்கம் உடைந்து பாட் விண்ணில் இருந்து கீழ்நோக்கி இறங்கியது. கீழே தெரிந்த பூமியைப் பார்த்த கதிரின் கண்கள் அதிர்ச்சியில் உறைந்து போயின. அங்கே கடல்களோ பசுமையான மரங்களோ இல்லை... முழு உலகமும் சாம்பல் நிற பாலைவனமாக மாறிப் போயிருந்தது! விண்வெளி வரை உயர்ந்த பிரம்மாண்டமான கறுப்பு உலோக கோபுரங்கள் வானத்தை விஷப் புகையால் மூடியிருந்தன. இந்த பேரழிவைத் தடுக்கத்தான் முதிய கதிர் தன் உயிரைக் கொடுத்தான் என்பதை உணர்ந்தான்.",
        "desc": "Jaya speaks with a suspenseful, sorrowful, and haunting storytelling tone with clear Tamil pronunciation in a close-sounding environment."
    },
    {
        "beat": 23,
        "name": "beat_23_citadel_of_chronos",
        "title": "Act VII: The Citadel of Chronos (காலக் கோட்டை)",
        "text": "பாட் ஒரு உடைந்த பாறைப் பள்ளத்தாக்கிற்குள் ரகசியமாகத் தரையிறங்கியது. கதிர் கையில் பிளாஸ்மா துப்பாக்கியோடு வெளியே குதித்தான். அவனுக்கு முன்னால் பல மைல் தூரத்திற்கு பரந்து விரிந்திருந்தது குரோனோஸ் கோட்டை! அந்த கோட்டையின் உச்சியில் தான் பிரதான கால இயந்திரத்தின் மையக் கரு இயங்கிக் கொண்டிருந்தது. அங்கிருந்து தான் முழு உலகையும் அழித்த விபரீத ஆற்றல் அலைகள் வெளிவந்து கொண்டிருந்தன.",
        "desc": "Jaya speaks with intense anticipation, steady pacing, and articulate Tamil diction in a close-sounding environment."
    },
    {
        "beat": 24,
        "name": "beat_24_infiltrating_the_reactor",
        "title": "Act VII: Infiltrating the Reactor (மையக் கருவுக்குள் நுழைவு)",
        "text": "கோட்டையின் பாதுகாப்பு ரேடார்கள் தன்னை அடையாளம் காணாதவாறு, மெமரி கிரிஸ்டலின் மறைமுக அலைவரிசையை கதிர் இயக்கினான். அவனது உடல் ஒரு மாய நிழல் போல மறைந்தது. கோட்டையின் பிரம்மாண்ட வாயில்களைக் கடந்து, ஆயிரக்கணக்கான இயந்திர ரோபோக்களின் கண்களில் படாமல் பிரதான அணுக்கரு உலைக்குள் பாய்ந்து சென்றான். அங்கு நின்றிருந்த மெகா ரியாக்டர் ஒரு நீல நிற சூரியனைப் போல கர்ஜித்துக் கொண்டிருந்தது.",
        "desc": "Jaya delivers a stealthy, tense, and thrilling narration in natural conversational Tamil with focused urgency."
    },
    {
        "beat": 25,
        "name": "beat_25_the_master_architect",
        "title": "Act VIII: Twist 5 - The Master Architect (மர்ம சூத்திரதாரி: அதிர்ச்சி உண்மை!)",
        "text": "அப்போது அந்த அறையில் விளக்குகள் பிரகாசமாக எரிந்தன. கட்டுப்பாட்டு மேடையில் இருந்து ஒரு கம்பீரமான நபர் கைதட்டியவாறு முன்னே வந்தார். கதிர் துப்பாக்கியை நீட்டினான்: யார் நீ? அந்த நபர் தன் முகமூடியைக் கழற்றினார். கதிரின் இதயம் நின்றுவிட்டது போல அதிர்ந்தான்! அங்கே நின்றிருந்தது அவனது கல்லூரித் தோழன் சித்தார்த்! முந்நூறு வருடங்கள் கழித்து ஒரு மனித சைபார்க்காக அவன் மாறி நின்றான்!",
        "desc": "Jaya speaks with dramatic shock, gasping disbelief, and expressive intensity in clear Tamil pronunciation."
    },
    {
        "beat": 26,
        "name": "beat_26_the_secret_conspiracy",
        "title": "Act VIII: The Secret Conspiracy (சித்தார்த்தின் வஞ்சகத் திட்டம்)",
        "text": "சித்தார்த் வஞ்சக சிரிப்போடு பேசினான்: ஆச்சரியப்படாதே கதிர்! உன்னால் ஒரு கால இயந்திரத்தை கண்டுபிடிக்க முடியும் என்றால், அதை வைத்து காலத்தையே என் கட்டுப்பாட்டுக்குள் கொண்டுவர என்னால் முடியாதா? கடந்த காலத்திற்கு முதிய கதிரை அனுப்பி, அங்கே ஒரு விபத்தை ஏற்படுத்தி, உன் முழு ஆராய்ச்சிகளையும் திருட நான்தான் அந்த குரோனோ ஹண்டர்களை அனுப்பினேன்! முழு காலத்தின் சக்கரமும் இப்போது என் கையில்!",
        "desc": "Jaya delivers cold, sinister, and villainous character dialogue with sharp, dramatic Tamil articulation."
    },
    {
        "beat": 27,
        "name": "beat_27_duel_of_destinies",
        "title": "Act IX: Duel of Destinies (விதியின் யுத்தம்)",
        "text": "கதிர் கோபத்தோடு கத்தினான்: ஒருபோதும் என் உலகத்தை அழிக்க உன்னை விடமாட்டேன்டா! கதிர் தன் பிளாஸ்மா துப்பாக்கியால் சுடத் தொடங்கினான். சித்தார்த் தன் இயந்திரக் கைகளில் இருந்து எலக்ட்ரிக் கத்திகளை உருவி கதிரை நோக்கி மின்னல் வேகத்தில் பாய்ந்தான். இருவருக்கும் இடையே அணுக்கரு உலையின் பாலத்தில் ஒரு கொடூரமான வாழ்வா சாவா யுத்தம் மூண்டது. துப்பாக்கிச் சத்தங்களும் உலோக மோதல்களும் அந்த மாளிகையை அதிரச் செய்தன.",
        "desc": "Jaya delivers high-octane battle action narration with breathless rhythm, forceful energy, and clear Tamil diction."
    },
    {
        "beat": 28,
        "name": "beat_28_overloading_the_core",
        "title": "Act IX: Overloading the Core (மையக் கருவின் வெடிப்பு)",
        "text": "ஒரு கடுமையான மோதலில் சித்தார்த் கதிரின் தோளில் காயத்தை ஏற்படுத்தினான். ஆனால் கதிர் வலியைப் பொருட்படுத்தாமல், தன் கையில் இருந்த நீல மெமரி கிரிஸ்டலை அந்த பிரதான அணுக்கரு ரியாக்டரின் மையத்திற்குள் வீசி எறிந்தான்! அந்த கிரிஸ்டல் ரியாக்டரின் அத்தனை ஆற்றலையும் தலைகீழாக மாற்றியது. கணினி அலறியது: பேரழிவு பிழை! கால விசை எதிர்மாறாக சுழல்கிறது! கோட்டை முழுதும் வெடித்துச் சிதற ஆரம்பித்தது. சித்தார்த் பயத்தில் அலறினான்: என்ன காரியம் செய்தாய் கதிர்?!",
        "desc": "Jaya speaks with explosive climax energy, escalating panic, and thrilling narrative momentum."
    },
    {
        "beat": 29,
        "name": "beat_29_the_temporal_reset",
        "title": "Act X: The Temporal Reset (காலத்தின் மறுபிறப்பு)",
        "text": "கதிர் கம்பீரமாகச் சொன்னான்: காலத்தை மாற்றியமைக்கவில்லை... உன்னைப்போன்ற பேராசை பிடித்தவர்களின் கைகளில் இருந்து அதை விடுவிக்கிறேன்! ஒரு மாபெரும் வெள்ளொளி முழு உலகையும் சூழ்ந்தது. இருநூற்று ஐம்பது வருட காலக்கோடு ஒரு கனவு போல சுருங்கி மறைந்தது. கோட்டையும் சாம்பல் உலகமும் மறைந்து, காலம் பின்னோக்கி ஓட ஆரம்பித்தது. கதிர் அந்த ஒளியின் நடுவே மெல்ல கண்களை மூடினான்.",
        "desc": "Jaya narrates the monumental cosmic reset with profound spiritual awe, calm authority, and melodic Tamil grace."
    },
    {
        "beat": 30,
        "name": "beat_30_grand_resolution_peaceful_dawn",
        "title": "Act X: Grand Resolution - The Peaceful Dawn (முடிவுரை: அமைதியான விடியல்)",
        "text": "கதிர் மெதுவாக கண்களைத் திறந்தான். குளிர்ந்த காலைக் காற்று அவன் முகத்தில் வீசியது. பறவைகளின் இனிய சத்தம் காதுகளில் கேட்டது. அவன் அதே மலை உச்சியில் நின்றுகொண்டிருந்தான்! வானத்தில் எந்த விண்கலமும் இல்லை, எந்த அழிவும் இல்லை. கிழக்கு வானத்தில் புதிய சூரியன் அழகாக உதயமாகிக் கொண்டிருந்தது. ஆனால் அவனது கையில் ஒரு சிறிய நீல கிரிஸ்டல் மெல்லிய ஒளியோடு மின்னியது. கதிர் புன்னகையோடு சொன்னான்: உலகம் காப்பாற்றப்பட்டுவிட்டது... புதிய விடியல் ஆரம்பமாகிவிட்டது!",
        "desc": "Jaya delivers a peaceful, triumphant, and deeply heartwarming resolution in pristine Tamil with resonant beauty."
    }
]

MODEL_REPO = "RXD03/indic-parler-tts"
models_by_dev = {}
tokenizers_by_dev = {}
desc_tokenizers_by_dev = {}

for dev in devices:
    print(f"[*] Initializing Indic Parler-TTS on {dev}...", flush=True)
    m = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float16 if "cuda" in dev else torch.float32
    ).to(dev)
    t = AutoTokenizer.from_pretrained(MODEL_REPO)
    dt = AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)
    models_by_dev[dev] = m
    tokenizers_by_dev[dev] = t
    desc_tokenizers_by_dev[dev] = dt
    print(f"[+] Device {dev} Ready! Sample Rate: {m.config.sampling_rate} Hz", flush=True)

SAMPLE_RATE = models_by_dev[devices[0]].config.sampling_rate

def synthesize_beat(beat_item, dev_str):
    m = models_by_dev[dev_str]
    t = tokenizers_by_dev[dev_str]
    dt = desc_tokenizers_by_dev[dev_str]
    
    b_num = beat_item["beat"]
    b_name = beat_item["name"]
    b_title = beat_item["title"]
    b_text = beat_item["text"]
    b_desc = beat_item["desc"]
    words = len(b_text.split())
    
    t0 = time.time()
    desc_inputs = dt(b_desc, return_tensors="pt").to(dev_str)
    prompt_inputs = t(b_text, return_tensors="pt").to(dev_str)
    
    with torch.inference_mode():
        gen = m.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
            do_sample=True,
            temperature=1.0
        )
        
    audio_np = gen.cpu().numpy().squeeze().astype(np.float32)
    dur_sec = round(len(audio_np) / SAMPLE_RATE, 2)
    elapsed = round(time.time() - t0, 2)
    
    peak = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0
    if peak > 0.98:
        audio_np = audio_np * (0.95 / peak)
        
    filename = f"{b_name}.wav"
    filepath = output_dir / filename
    sf.write(str(filepath), audio_np, SAMPLE_RATE, subtype="PCM_16")
    
    wpm = int((words / max(dur_sec, 0.001)) * 60)
    print(f"[{dev_str}] Beat {b_num:02d}/30 ({b_name}): {dur_sec}s | Compute: {elapsed}s | Cadence: {wpm} WPM", flush=True)
    
    del gen, desc_inputs, prompt_inputs
    gc.collect()
    if "cuda" in dev_str:
        torch.cuda.empty_cache()
        
    return {
        "beat": b_num,
        "name": b_name,
        "title": b_title,
        "filename": filename,
        "duration_sec": dur_sec,
        "words": words,
        "wpm": wpm,
        "compute_sec": elapsed
    }

processed_records = []
if len(devices) > 1:
    print(f"[*] Running Dual-GPU concurrent synthesis for Beats 21-30...", flush=True)
    mid = len(BEATS_PART2) // 2
    groups = [
        (BEATS_PART2[:mid], devices[0]),
        (BEATS_PART2[mid:], devices[1])
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for subset, dev in groups:
            for b in subset:
                futures.append(executor.submit(synthesize_beat, b, dev))
        for f in futures:
            processed_records.append(f.result())
else:
    dev = devices[0]
    for b in BEATS_PART2:
        processed_records.append(synthesize_beat(b, dev))

processed_records.sort(key=lambda x: x["beat"])

with open(output_dir / "part2_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "status": "success",
        "total_beats": len(processed_records),
        "total_duration_sec": sum(r["duration_sec"] for r in processed_records),
        "records": processed_records
    }, f, ensure_ascii=False, indent=2)

print("[+] Part 2 Synthesis Completed Successfully!", flush=True)
