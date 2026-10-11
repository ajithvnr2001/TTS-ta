import os
import sys
import json
import glob
from pathlib import Path
import soundfile as sf
import numpy as np

def stitch_master():
    print("=" * 80)
    print("STITCHING DEFINITIVE 10-MINUTE CINEMATIC THRILLER MASTER AUDIO")
    print("=" * 80)

    part1_dir = Path("D:/azure/indic-parler-tts-kaggle/output_10min/output_10min")
    if not part1_dir.exists():
        part1_dir = Path("D:/azure/indic-parler-tts-kaggle/output_10min")
    
    part2_dir = Path("D:/azure/indic-parler-tts-kaggle/output_part2/output_part2")
    if not part2_dir.exists():
        part2_dir = Path("D:/azure/indic-parler-tts-kaggle/output_part2")

    print(f"[*] Part 1 directory: {part1_dir}")
    print(f"[*] Part 2 directory: {part2_dir}")

    # Gather Part 1 files: beat_01 to beat_20
    beats = {}
    for p in part1_dir.glob("beat_*.wav"):
        fname = p.name
        # format: beat_01_name.wav
        beat_num = int(fname.split("_")[1])
        beats[beat_num] = p

    # Gather Part 2 files: beat_21 to beat_30
    for p in part2_dir.glob("beat_*.wav"):
        fname = p.name
        beat_num = int(fname.split("_")[1])
        beats[beat_num] = p

    print(f"[+] Total beats found: {len(beats)}")
    for i in sorted(beats.keys()):
        print(f"  Beat {i:02d}: {beats[i].name}")

    if len(beats) < 30:
        print(f"[!] Warning: Only found {len(beats)} beats (expected 30). Checking what is missing...")
        missing = [i for i in range(1, 31) if i not in beats]
        print(f"    Missing beats: {missing}")
        if len(beats) == 0:
            return False

    all_audio = []
    sr = 44100
    silence = np.zeros(int(sr * 0.5), dtype=np.float32) # 0.5s pause
    
    timeline = []
    current_time = 0.0

    beat_titles = {
        1: "Act I - Midnight Anomaly (நள்ளிரவு மர்மம்)",
        2: "Act I - Descent into Crater (பள்ளத்தாக்கின் மர்மம்)",
        3: "Act I - Breached Airlock (திறந்த கதவு)",
        4: "Act I - Twist 1: Earth Coordinates (முதல் திருப்பம்: பூமி தான் அது!)",
        5: "Act II - The Groaning Cockpit (விண்கலத்தின் அழுகுரல்)",
        6: "Act II - Twist 2: Unmasking the Pilot (இரண்டாம் திருப்பம்: அதே முகம், அதே தழும்பு!)",
        7: "Act II - The Apocalypse Warning (காலப் பேரழிவு எச்சரிக்கை)",
        8: "Act II - Twist 3: The Fatal Paradox (மூன்றாம் திருப்பம்: நீ சாகவில்லை, சிக்கியுள்ளாய்!)",
        9: "Act III - The Countdown Arms (இடி மின்னல் கவுண்டவுன்)",
        10: "Act III - Twist 4: Chrono Hunters Arrive (நான்காம் திருப்பம்: நிழல் வேட்டைக்காரர்கள்!)",
        11: "Act III - The Forest Encirclement (சுற்றுவளைக்கப்பட்ட காடு)",
        12: "Act III - Twist 5: The EMP Blast (ஐந்தாம் திருப்பம்: மின்காந்த அதிர்ச்சி அலை!)",
        13: "Act IV - The Memory Crystal (நினைவுப் படிகத்தின் ரகசியம்)",
        14: "Act IV - Twist 6: The Terrible Cost (ஆறாம் திருப்பம்: ஒரு காலக்கோடு அழிய வேண்டும்!)",
        15: "Act IV - The Hull Breach (உடைபட்ட விண்கலக் கூடு)",
        16: "Act IV - Twist 7: The Last Stand (ஏழாம் திருப்பம்: உனக்காக நான் நிற்கிறேன்!)",
        17: "Act V - The Escape Pod (தப்புவிக்கும் காப்ஸ்யூல்)",
        18: "Act V - Twist 8: Sealing the Hatch (எட்டாம் திருப்பம்: விடைபெறும் மூத்த கதிர்!)",
        19: "Act V - The Cobalt Explosion (நீலக் கோபால்ட் வெடிப்பு)",
        20: "Act V - Twist 9: The Loop Shattered (ஒன்பதாம் திருப்பம்: அறுபட்ட காலச் சங்கிலி!)",
        21: "Act VI - Earth 2550 (பூமி 2550: கருகிய எதிர்காலம்)",
        22: "Act VI - Twist 10: The Citadel of Chronos (பத்தாம் திருப்பம்: காலப் பெருங்கோபுரம்!)",
        23: "Act VII - Infiltration (கொடூர ரோபோக்களின் அரண்மனைக்குள்)",
        24: "Act VII - Twist 11: The Secret Core (பதினொன்றாம் திருப்பம்: பிரபஞ்சத்தின் இதயத் துடிப்பு!)",
        25: "Act VIII - The Master Architect (காலப் பெருஞ்சிற்பி சித்தார்த்)",
        26: "Act VIII - Twist 12: The Architect Identity (பன்னிரண்டாம் திருப்பம்: காலத்தை அடிமையாக்கியவன்!)",
        27: "Act IX - The Duel of Timelines (காலக் கோடுகளின் இறுதி மோதல்)",
        28: "Act IX - Twist 13: Overloading Core (பதின்மூன்றாம் திருப்பம்: பேரழிவை வெடிக்கும் தியாகம்!)",
        29: "Act X - The Great Collapse (நொறுங்கி விழும் காலச் சிறை)",
        30: "Act X - Twist 14: The Peaceful Dawn (இறுதித் திருப்பம்: மலர்ந்த அமைதி விடியல்!)"
    }

    total_words = 0
    for beat_num in sorted(beats.keys()):
        p = beats[beat_num]
        audio, file_sr = sf.read(str(p))
        if file_sr != sr:
            import resampy
            audio = resampy.resample(audio, file_sr, sr)
        
        # ensure 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        dur = len(audio) / sr
        start_t = current_time
        end_t = start_t + dur

        timeline.append({
            "beat": beat_num,
            "filename": p.name,
            "title": beat_titles.get(beat_num, f"Beat {beat_num}"),
            "start_time_sec": round(start_t, 2),
            "start_time_formatted": f"{int(start_t // 60):02d}:{int(start_t % 60):02d}",
            "end_time_sec": round(end_t, 2),
            "end_time_formatted": f"{int(end_t // 60):02d}:{int(end_t % 60):02d}",
            "duration_sec": round(dur, 2)
        })

        all_audio.append(audio)
        all_audio.append(silence) # 0.5s pause
        current_time = end_t + 0.5

    # Concat
    master_audio = np.concatenate(all_audio)
    
    # Peak normalize to 0.95
    max_val = np.max(np.abs(master_audio))
    if max_val > 0:
        master_audio = master_audio * (0.95 / max_val)

    total_duration_sec = len(master_audio) / sr
    total_duration_min = total_duration_sec / 60.0

    print("=" * 80)
    print(f"[+] Total Concatenated Duration: {total_duration_sec:.2f} seconds ({total_duration_min:.2f} minutes)")
    print(f"[+] Requirement >= 600.0s met? {'YES! PASSED' if total_duration_sec >= 600.0 else 'NO - STILL UNDER'}")
    print("=" * 80)

    # Save to both destinations
    out_path1 = Path("D:/azure/indic-parler-tts-kaggle/output_10min/tamil_10min_cinematic_thriller_master.wav")
    out_path2 = Path("D:/azure/indic-parler-tts-kaggle/tamil_10min_cinematic_thriller_master.wav")
    
    sf.write(str(out_path1), master_audio, sr, subtype="PCM_16")
    sf.write(str(out_path2), master_audio, sr, subtype="PCM_16")
    print(f"[+] Master WAV saved to {out_path1} ({out_path1.stat().st_size / (1024*1024):.2f} MB)")
    print(f"[+] Master WAV saved to {out_path2} ({out_path2.stat().st_size / (1024*1024):.2f} MB)")

    # Save complete summary JSON
    summary = {
        "status": "success",
        "project": "tamil_10min_cinematic_thriller_full",
        "title": "காலச் சுழல் (The Chrono Spiral) - முழுமையான 10 நிமிட பிரம்மாண்ட திரில்லர்",
        "narrator": "Jaya (AI4Bharat Indic Parler-TTS)",
        "target_reached_10mins": bool(total_duration_sec >= 600.0),
        "total_duration_sec": round(total_duration_sec, 2),
        "total_duration_min": round(total_duration_min, 2),
        "sample_rate_hz": sr,
        "format": "16-bit PCM Mono WAV, Peak-Normalized (0.95)",
        "total_beats": len(beats),
        "timeline": timeline
    }

    sum_path1 = Path("D:/azure/indic-parler-tts-kaggle/output_10min/thriller_10min_full_summary.json")
    sum_path2 = Path("D:/azure/indic-parler-tts-kaggle/thriller_10min_full_summary.json")
    
    with open(sum_path1, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(sum_path2, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[+] Full summary saved to {sum_path1} and {sum_path2}")
    return total_duration_sec >= 600.0

if __name__ == "__main__":
    stitch_master()
