"""
AI4Bharat Indic Parler-TTS: 5-Minute Sci-Fi Thriller Production Engine
Title: காலச் சுழல் (The Chrono Spiral) - விதியின் ரகசியம்
Features:
- Authentic Native Tamil Speech via IIT Madras & Hugging Face Indic Parler-TTS
- 10 Dynamic Story Beats with Sci-Fi Twists & Turns
- Continuous ~5-Minute Cinematic Master Audio (44.1 kHz, 16-bit PCM)
- Peak-Normalized, Timestamped Scene Transitions
"""

import os
import sys
import time
import json
import gc
import subprocess
from pathlib import Path

# Prevent transformers from importing tensorflow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Force unbuffered live stdout/stderr
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

print("=" * 75, flush=True)
print(" AI4BHARAT INDIC PARLER-TTS: 5-MINUTE TAMIL SCI-FI THRILLER ENGINE")
print(" Title: காலச் சுழல் (The Chrono Spiral) - விதியின் ரகசியம்")
print(" Speaker: Jaya (Native Conversational Tamil Narration)")
print(" Audio Standard: 44,100 Hz, 16-bit PCM Mono, Peak-Normalized")
print("=" * 75, flush=True)

# ------------------------------------------------------------------------------
# 1. DEPENDENCY INSTALLATION
# ------------------------------------------------------------------------------
def ensure_dependencies():
    print("[*] Installing required runtime dependencies...", flush=True)
    pkgs = [
        "protobuf>=5.26.1",
        "soundfile",
        "accelerate",
        "transformers>=4.44.0"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir"] + pkgs, check=True)

    print("[*] Installing parler-tts from Hugging Face official repository...", flush=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "git+https://github.com/huggingface/parler-tts.git"
    ], check=True)
    print("[+] Runtime dependencies installed successfully!", flush=True)

ensure_dependencies()

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# ------------------------------------------------------------------------------
# 2. MODEL INITIALIZATION (GPU)
# ------------------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[*] Initializing Indic Parler-TTS on device: {device}...", flush=True)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[*] GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)", flush=True)

candidate_repos = [
    "ai4bharat/indic-parler-tts",
    "RXD03/indic-parler-tts",
    "Yogi5/indic-parler-tts-backup"
]

model = None
tokenizer = None
description_tokenizer = None
loaded_repo = None

for repo in candidate_repos:
    try:
        print(f"[*] Attempting to load model from: {repo}...", flush=True)
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            repo,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(repo)
        text_encoder_path = model.config.text_encoder._name_or_path
        print(f"[*] Loading description tokenizer from: {text_encoder_path}...", flush=True)
        description_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
        loaded_repo = repo
        print(f"[+] Successfully loaded Indic Parler-TTS from: {loaded_repo}!", flush=True)
        break
    except Exception as e:
        print(f"[-] Failed loading {repo}: {e}. Trying next candidate...", flush=True)

if model is None:
    raise RuntimeError("[-] All candidate repositories failed to load Indic Parler-TTS.")

SAMPLE_RATE = model.config.sampling_rate
print(f"[+] Model Ready! Output Sample Rate: {SAMPLE_RATE} Hz", flush=True)

output_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# 3. 5-MINUTE EPIC SCI-FI THRILLER STORY BEATS (10 SCENES WITH TWISTS & TURNS)
# ------------------------------------------------------------------------------
THRILLER_STORY_BEATS = [
    {
        "beat": 1,
        "name": "beat_01_midnight_anomaly",
        "title": "The Midnight Anomaly (நள்ளிரவு மர்மம்)",
        "text": "அமைதியான ஒரு நள்ளிரவு நேரம். மலை உச்சிக்கு மேலே தனியாக நின்றுகொண்டிருந்த கதிர், வானத்தில் தெரிந்த விசித்திரமான நீல நிற ஒளியை வியப்போடு உற்றுப் பார்த்தான். அது சும்மா விண்கல் இல்லை... மேகங்களை கிழிச்சுக்கிட்டு காட்டுக்குள்ள மெதுவா இறங்கிய ஒரு பிரம்மாண்டமான உலோகப் பேழை! டேய், இது என்னடா இது? ஊருக்குள்ள யாருமே இதை கவனிக்கலையா? தனக்குள்ளேயே பேசிக்கிட்டு, இதயத் துடிப்பு படபடன்னு எகிற காட்டுக்குள்ள அவன் வேகமாக ஓடினான்.",
        "desc": "Jaya speaks in a calm, clear, and atmospheric narrative tone with natural Tamil cadence and subtle mystery in a close-sounding environment. Very high quality audio with no background noise."
    },
    {
        "beat": 2,
        "name": "beat_02_breached_hatch",
        "title": "The Breached Hatch (திறந்த கதவு)",
        "text": "கட்டடத்தின் அளவுக்கு பெரிய அந்த விண்கலம் மரங்களை வேரோடு உடைத்து தரையில் புதைஞ்சு போயிருந்தது. சுத்தி எந்த சத்தமும் இல்லை, ஆனா ஒரு நீல நிற கதிர்வீச்சு புகையோடு பரவுச்சு. கதிர் பக்கத்துல போனதும் அவனோட உடம்பே சிலிர்த்துச்சு. காரணம், விண்கலத்தின் கதவு தானாகவே திறந்துச்சு! உள்ளே இருந்து வந்த காற்று பனி மாதிரி சில்லுனு இருந்துச்சு. அந்த மினுமினுக்கும் நீல வெளிச்சம் அவனை உள்ளே வான்னு அழைச்சது.",
        "desc": "Jaya speaks with a suspenseful, intriguing, and hushed storytelling tone with clear Tamil pronunciation in a close-sounding environment. High audio quality."
    },
    {
        "beat": 3,
        "name": "beat_03_twist1_earth_coordinates",
        "title": "Twist 1: Earth Coordinates (முதல் திருப்பம்: பூமி தான் அது!)",
        "text": "உள்ளே போன கதிரோட கண்கள் ஆச்சரியத்துல விரிஞ்சு போச்சு. அங்கே எந்த ஒரு வேற்றுக்கிரக எழுத்துக்களும் இல்லை... கட்டுப்பாட்டுத் திரைகளில் ஓடியது அத்தனையும் தமிழ் எழுத்துக்களும் பூமி எண்களும் தான்! நடுங்கும் கைகளோடு கதிர் அந்த தொடுதிரையை தொட்டான். ஒரு கணினி குரல் கம்பீரமாக ஒலிச்சது: விண்கலம் காலப் பயணம் தொகுதி நான்கு. புறப்பட்ட இடம் பூமி. ஆண்டு இரண்டாயிரத்து ஐந்நூற்று ஐம்பது! கதிர் நடுங்கிப் போனான்... என்னது? இது வேற்றுக்கிரக கப்பல் இல்லையா? எதிர்கால பூமியில இருந்து வந்திருக்கா?!",
        "desc": "Jaya speaks with expressive shock and dramatic realization, delivering clear Tamil diction with natural emotional intensity."
    },
    {
        "beat": 4,
        "name": "beat_04_twist2_familiar_pilot",
        "title": "Twist 2: The Familiar Pilot (இரண்டாவது திருப்பம்: அவன் குரல்!)",
        "text": "அப்போ, காக்பிட் அறைக்குள்ள இருந்து பலவீனமான ஒரு இருமல் சத்தம் கேட்டுச்சு. கதிர் அதிர்ச்சியோடு மெதுவா எட்டிப் பார்த்தான். பைலட் இருக்கையில விண்வெளி உடையில் ஒருவர் ரத்தக் காயங்களோடு மயங்கி சாய்ஞ்சு கிடந்தார். அந்த நபர் நடுங்கும் கைகளால் தன் தலைக்கவசத்தை கழட்டினார். கதிரோட மூச்சே நின்னு போச்சு! அங்கே உட்கார்ந்திருந்தது வேற யாரும் இல்லை... முப்பது வருடம் மூத்த கதிர்! அவனோட அதே நெற்றித் தழும்பு, அதே தீவிர கண்கள்!",
        "desc": "Jaya speaks with breathless disbelief and emotional awe, maintaining crisp Tamil pronunciation and realistic pacing."
    },
    {
        "beat": 5,
        "name": "beat_05_temporal_warning",
        "title": "The Temporal Warning (எதிர்கால எச்சரிக்கை)",
        "text": "முதிய கதிர் அவனைப் பார்த்து ஒரு பலவீனமான சிரிப்போடு பேசினான்: கதிர்... கரெக்டா இந்த நைட்டுக்காகத்தான் நான் முந்நூறு வருஷமா காத்துக்கிட்டு இருந்தேன். நீ இப்போ பார்க்குற எல்லாமே உண்மைடா. நான் உன்னோட எதிர்காலம்! பூமி அழியப் போகுதுன்னு தெரிஞ்சு, அதைத் தடுத்து நிறுத்த காலத்தை பின்னோக்கி சுழற்றி இங்க வந்தேன். இளைய கதிர் பதற்றத்தோடு கேட்டான்: நீ... நானா? அப்போ நான் என்ன பண்ணனும்?",
        "desc": "Jaya delivers poignant, grave, and intense character dialogue in authentic Tamil, with clear and emotional vocal control."
    },
    {
        "beat": 6,
        "name": "beat_06_twist3_trap_and_timer",
        "title": "Twist 3: The Trap & Timer (மூன்றாவது திருப்பம்: மரணப் பொறி!)",
        "text": "முதிய கதிர் அவன் கையை இறுக்கமாகப் பிடித்தான்: கதிர், நாம உலகத்தைக் காப்பாத்த வரல... ஒரு பயங்கரமான சதியை தெரிஞ்சிக்கிட்டு வந்தேன். எதிர்காலத்துல நாம கண்டுபிடிச்ச இந்த கால இயந்திரம் தான் பூமியை அழிக்கப் போற மூல காரணமே! நீ உள்ளே காலடி வச்ச அடுத்த நொடி, இந்த விண்கலத்தோட ஆட்டோ ரீசெட் டைமர் ஆன் ஆயிடுச்சு. இன்னும் பத்து நிமிடத்துல இந்த மலைப்பகுதியே வெடிச்சு சாம்பலாகப் போகுது!",
        "desc": "Jaya speaks with urgent alarm and rapid dramatic intensity, delivering articulate Tamil words with gripping momentum."
    },
    {
        "beat": 7,
        "name": "beat_07_twist4_chrono_hunters",
        "title": "Twist 4: The Chrono Hunters (நான்காவது திருப்பம்: கால வேட்டையர்கள்!)",
        "text": "என்ன சொல்ற நீ? அப்போ நாம ரெண்டு பேருமே இங்கிருந்து தப்பிக்க முடியாதா?! கதிர் அலறினான். முதிய கதிர் நடுங்கும் விரல்களால் கட்டுப்பாட்டுத் திரையைக் காட்டினான்: தப்பிக்கலாம்... ஆனா நம்ம பின்னாடியே காலக்காவலர்கள், அதாவது குரோனோ ஹண்டர்ஸ் காட்டுக்குள்ள தேடி வந்துட்டாங்க. ஜன்னலை பாருடா! கதிர் திரையைப் பார்த்தான். அடர்ந்த காட்டு மரங்களுக்கு நடுவே நூற்றுக்கணக்கான சிவப்பு நிற ட்ரோன்கள் ஒளிரும் கொடூர கண்களோடு விண்கலத்தை சுற்றி வளைச்சு நின்னது!",
        "desc": "Jaya speaks with heightened adrenaline, panic, and fast narrative rhythm with authentic Tamil exclamation and cadence."
    },
    {
        "beat": 8,
        "name": "beat_08_memory_crystal",
        "title": "The Memory Crystal (நினைவுக் கிரிஸ்டல்)",
        "text": "இப்போ தப்பிக்க ஒரே ஒரு வழி தான் இருக்கு! முதிய கதிர் தன் சட்டைப் பையில இருந்து ஒரு நீல நிற கிரிஸ்டலை எடுத்தான். இந்த மெமரி கோர்... இதுல எதிர்காலத்துல நாம பண்ண எல்லா தப்புகளோட ரகசிய டேட்டாவும் இருக்கு. இதை உன் உடம்புல பொருத்தினா, உனக்கு எதிர்கால ஞாபகங்கள் அத்தனையும் கிடைக்கும். ஆனா ஒரு ஆபத்து... நீ பழைய கதிராக இருக்க முடியாது. காலச் சுழலுக்குள்ள நீயும் சிக்கிக்குவ!",
        "desc": "Jaya delivers a mystical, intense, and consequential scientific exposition with clear and steady Tamil articulation."
    },
    {
        "beat": 9,
        "name": "beat_09_ultimate_sacrifice",
        "title": "The Ultimate Sacrifice (இறுதி தியாகம்)",
        "text": "ட்ரோன்கள் விண்கலத்தின் உலோகக் கதவை வெடிகுண்டுகளால் உடைக்க ஆரம்பிச்சது. லேசர் தோட்டாக்கள் மழை மாதிரி பொழிந்தது. முதிய கதிர் அவசர அவசரமாக எஸ்கேப் பாட் கதவை திறந்தான். கதிர், என் காலம் இதோட முடிஞ்சு போச்சு... ஆனா நீ உயிரோட தப்பிச்சாகணும்! இந்த கிரிஸ்டலை வச்சு காலத்தின் கொடூர லூப்பை நீ உடைக்கணும்டா. இந்த தடவை நீ கண்டிப்பா ஜெயிக்கணும்! முதிய கதிர் அவனை எஸ்கேப் பாட்டுக்குள் தள்ளி, பாதுகாப்பு பட்டனை அழுத்தினான்.",
        "desc": "Jaya speaks with heroic emotion, sacrifice, and dramatic urgency in powerful Tamil expression."
    },
    {
        "beat": 10,
        "name": "beat_10_climax_loop_shattered",
        "title": "Climax: Loop Shattered (முடிவுரை: காலச் சுழலின் ஆரம்பம்!)",
        "text": "எஸ்கேப் பாட் விண்ணை நோக்கி மின்னல் வேகத்தில் சீறிப் பாய்ந்தது. கீழே மலை உச்சியில் அந்த விண்கலம் ஒரு மகா வெடிப்போடு நீல நிற ஒளியில் சிதறியது! பாட்டின் உள்ளே இருந்த கதிர், தன் கையில் இருந்த நீல கிரிஸ்டலை நெஞ்சில் பொருத்தினான். அவனது கண்களில் ஒரு புதிய எதிர்கால வெளிச்சம் பிறந்தது. கணினி திரை கம்பீரமாக ஒலித்தது: வெல்கம் பேக் கேப்டன் கதிர்! ஒரு நம்பிக்கையான புன்னகையோடு கதிர் சொன்னான்: ஆட்டத்தை இப்பதான் நான் ஆரம்பிக்கிறேன்!",
        "desc": "Jaya speaks with cinematic triumph, energy, and resolute confidence, delivering a thrilling climax with clear Tamil cadence."
    }
]

# ------------------------------------------------------------------------------
# 4. BATCH GENERATION EXECUTION
# ------------------------------------------------------------------------------
print(f"\n[*] Starting 5-Minute Story Synthesis across {len(THRILLER_STORY_BEATS)} story beats...", flush=True)

audio_segments = []
metadata_records = []
t_total_start = time.time()
current_timeline_sec = 0.0

for idx, beat in enumerate(THRILLER_STORY_BEATS):
    b_num = beat["beat"]
    b_name = beat["name"]
    b_title = beat["title"]
    b_text = beat["text"]
    b_desc = beat["desc"]
    words = b_text.split()
    word_count = len(words)
    
    print(f"\n{'='*70}", flush=True)
    print(f"[+] Processing Beat {b_num:02d}/{len(THRILLER_STORY_BEATS)}: {b_title}", flush=True)
    print(f"    Words: {word_count} | Timeline Start: {round(current_timeline_sec, 2)}s", flush=True)
    print(f"    Text: {b_text[:80]}...", flush=True)
    print(f"    Conditioning: {b_desc[:90]}...", flush=True)
    
    t_beat_start = time.time()
    
    desc_inputs = description_tokenizer(b_desc, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(b_text, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        generation = model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
            do_sample=True,
            temperature=1.0
        )
        
    audio_np = generation.cpu().numpy().squeeze().astype(np.float32)
    dur_sec = round(len(audio_np) / SAMPLE_RATE, 2)
    elapsed_sec = round(time.time() - t_beat_start, 2)
    rtf = round(elapsed_sec / max(dur_sec, 0.001), 3)
    w_sec = round(word_count / max(dur_sec, 0.001), 2)
    wpm = int(w_sec * 60)
    
    print(f"    -> Duration: {dur_sec}s | Compute: {elapsed_sec}s | RTF: {rtf} | Cadence: {wpm} WPM", flush=True)
    
    # Peak normalization to prevent clipping
    peak_val = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0
    if peak_val > 0.98:
        audio_np = audio_np * (0.95 / peak_val)
        
    beat_filename = f"{b_name}.wav"
    beat_filepath = output_dir / beat_filename
    sf.write(str(beat_filepath), audio_np, SAMPLE_RATE, subtype="PCM_16")
    
    audio_segments.append(audio_np)
    
    metadata_records.append({
        "beat": b_num,
        "name": b_name,
        "title": b_title,
        "filename": beat_filename,
        "timeline_start_sec": round(current_timeline_sec, 2),
        "duration_sec": dur_sec,
        "timeline_end_sec": round(current_timeline_sec + dur_sec, 2),
        "words": word_count,
        "wpm": wpm,
        "compute_sec": elapsed_sec,
        "rtf": rtf
    })
    
    current_timeline_sec += dur_sec + 0.4  # include 0.4s inter-scene transition
    
    del generation, desc_inputs, prompt_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ------------------------------------------------------------------------------
# 5. MASTER TRACK CONCATENATION (WITH CINEMATIC SCENE PAUSES)
# ------------------------------------------------------------------------------
print(f"\n{'='*70}", flush=True)
print("[*] Stitching 5-minute continuous cinematic master audio...", flush=True)

pause_samples = int(0.4 * SAMPLE_RATE)
silence_np = np.zeros(pause_samples, dtype=np.float32)

stitched_segments = []
for idx, seg in enumerate(audio_segments):
    stitched_segments.append(seg)
    if idx < len(audio_segments) - 1:
        stitched_segments.append(silence_np)

master_np = np.concatenate(stitched_segments)
peak_master = float(np.max(np.abs(master_np))) if len(master_np) > 0 else 0.0
if peak_master > 0.98:
    master_np = master_np * (0.95 / peak_master)

master_filename = "tamil_5min_thriller_master.wav"
master_filepath = output_dir / master_filename
sf.write(str(master_filepath), master_np, SAMPLE_RATE, subtype="PCM_16")

master_duration_sec = round(len(master_np) / SAMPLE_RATE, 2)
master_duration_min = round(master_duration_sec / 60.0, 2)
total_compute_sec = round(time.time() - t_total_start, 2)

print(f"[+] Master Audio generated successfully!", flush=True)
print(f"    File: {master_filename}", flush=True)
print(f"    Duration: {master_duration_sec}s ({master_duration_min} minutes)", flush=True)
print(f"    Sample Rate: {SAMPLE_RATE} Hz (16-bit PCM)", flush=True)
print(f"    Total Synthesis Time: {total_compute_sec}s", flush=True)

# ------------------------------------------------------------------------------
# 6. PRODUCTION METADATA REPORT
# ------------------------------------------------------------------------------
production_summary = {
    "status": "success",
    "project": "tamil_5min_scifi_thriller",
    "title": "காலச் சுழல் (The Chrono Spiral) - விதியின் ரகசியம்",
    "speaker": "Jaya (AI4Bharat Indic Parler-TTS)",
    "institution": "AI4Bharat (IIT Madras) & Hugging Face",
    "model_repo": loaded_repo,
    "sample_rate_hz": SAMPLE_RATE,
    "channels": 1,
    "format": "WAV (PCM_16)",
    "master_filename": master_filename,
    "master_duration_sec": master_duration_sec,
    "master_duration_min": master_duration_min,
    "total_words": sum(b["words"] for b in metadata_records),
    "total_beats": len(metadata_records),
    "total_compute_sec": total_compute_sec,
    "beats": metadata_records
}

summary_path = output_dir / "thriller_5min_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(production_summary, f, ensure_ascii=False, indent=2)

print(f"[+] Production Summary saved to: {summary_path}", flush=True)
print("=" * 75, flush=True)
print(" 5-MINUTE TAMIL SCI-FI THRILLER SYNTHESIS COMPLETED SUCCESSFULLY!")
print("=" * 75, flush=True)
