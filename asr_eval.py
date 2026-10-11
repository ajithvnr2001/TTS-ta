import soundfile as sf
import speech_recognition as sr
import os

r = sr.Recognizer()

beats_to_test = {
    'indic_parler_jaya_beat_1': r'D:\azure\indic-parler-tts-kaggle\output\indic_parler_jaya_beat_1.wav',
    'indic_parler_jaya_beat_2': r'D:\azure\indic-parler-tts-kaggle\output\indic_parler_jaya_beat_2.wav',
    'indic_parler_jaya_beat_3': r'D:\azure\indic-parler-tts-kaggle\output\indic_parler_jaya_beat_3.wav',
    'indic_parler_jaya_beat_4': r'D:\azure\indic-parler-tts-kaggle\output\indic_parler_jaya_beat_4.wav',
    'indic_parler_kavitha_beat_1': r'D:\azure\indic-parler-tts-kaggle\output\indic_parler_kavitha_beat_1.wav',
    'indic_parler_kavitha_beat_2': r'D:\azure\indic-parler-tts-kaggle\output\indic_parler_kavitha_beat_2.wav',
    'indic_parler_kavitha_beat_3': r'D:\azure\indic-parler-tts-kaggle\output\indic_parler_kavitha_beat_3.wav',
    'indic_parler_kavitha_beat_4': r'D:\azure\indic-parler-tts-kaggle\output\indic_parler_kavitha_beat_4.wav',
    'fish_v12_script_master': r'D:\azure\fish-tts-kaggle\tamil_audio\tamil_master_script_v12.wav',
    'indicvoices_sample_1_Salem': r'D:\azure\indic-parler-tts-kaggle\indicvoices_samples\indicvoices_sample_1_0_Salem.wav'
}

results = []

for name, p in beats_to_test.items():
    if not os.path.exists(p):
        continue
    data, sr_hz = sf.read(p)
    dur = round(len(data)/sr_hz, 2)
    with sr.AudioFile(p) as source:
        audio_data = r.record(source)
        try:
            transcription = r.recognize_google(audio_data, language='ta-IN')
            status = 'OK'
        except sr.UnknownValueError:
            transcription = '[Unrecognized / Unintelligible]'
            status = 'FAILED'
        except Exception as e:
            transcription = f'[Error: {e}]'
            status = 'ERROR'
            
    results.append({
        'name': name,
        'duration_sec': dur,
        'sample_rate_hz': sr_hz,
        'transcription': transcription,
        'status': status
    })

out_txt = r'D:\azure\indic-parler-tts-kaggle\output\asr_evaluation_report.txt'
with open(out_txt, 'w', encoding='utf-8') as f:
    f.write("=== TAMIL ASR RECOGNITION EVALUATION REPORT (ta-IN) ===\n\n")
    for item in results:
        f.write(f"[{item['name']}] (Duration: {item['duration_sec']}s, SR: {item['sample_rate_hz']}Hz, Status: {item['status']})\n")
        f.write(f"  Transcription: {item['transcription']}\n\n")

print(f"ASR report written successfully to {out_txt}")
