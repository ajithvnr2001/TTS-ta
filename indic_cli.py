"""
Indic Parler-TTS Unified CLI Tool
Programmatic, End-to-End Command-Line Interface for Tamil TTS Synthesis on Kaggle GPU
AI4Bharat (IIT Madras) Native Indian Speech Architecture
No Cloudflare tunnel, No WebUI - Pure CLI & Batch Programmatic Pipeline.
"""

import os
import sys
import time
import json
import wave
import struct
import math
import argparse
import subprocess
from pathlib import Path

KERNEL_ID = "ajithkumaevnr2001/indic-parler-tts"
WORKSPACE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = WORKSPACE_DIR / "tamil_audio"

def run_cmd(cmd_list, capture=True):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd_list, capture_output=capture, text=True, env=env, cwd=str(WORKSPACE_DIR))
    return res.returncode, res.stdout, res.stderr

def cmd_push():
    print(f"[*] Pushing Kaggle kernel for Indic Parler-TTS ({KERNEL_ID})...")
    code, stdout, stderr = run_cmd(["kaggle", "kernels", "push", "-p", str(WORKSPACE_DIR)])
    print(stdout.strip())
    if code != 0:
        print(f"[-] Push failed: {stderr.strip()}")
        sys.exit(code)
    print("[+] Kernel pushed successfully!")

def cmd_status():
    code, stdout, stderr = run_cmd(["kaggle", "kernels", "status", KERNEL_ID])
    status_str = stdout.strip()
    print(f"[*] Current Kernel Status: {status_str}")
    return status_str

def cmd_logs():
    print(f"[*] Fetching Kaggle kernel logs for {KERNEL_ID}...")
    code, stdout, stderr = run_cmd(["kaggle", "kernels", "logs", KERNEL_ID])
    if stdout.strip():
        try:
            entries = json.loads(stdout)
            for entry in entries:
                stream = entry.get("stream_name", "out")
                data = entry.get("data", "")
                sys.stdout.write(f"[{stream}] {data}")
        except Exception:
            print(stdout)
    else:
        print("[*] No logs yet or log empty.")
    if stderr.strip():
        print(f"[STDERR] {stderr.strip()}")

def cmd_download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading generated audio outputs from Kaggle to {OUTPUT_DIR}...")
    code, stdout, stderr = run_cmd(["kaggle", "kernels", "output", KERNEL_ID, "-p", str(OUTPUT_DIR), "-o"])
    print(stdout.strip())
    if stderr.strip():
        print(f"[-] Note: {stderr.strip()}")
    
    files = list(OUTPUT_DIR.glob("*.wav")) + list(OUTPUT_DIR.glob("*.json"))
    print(f"[+] Download complete. Found {len(files)} output assets in {OUTPUT_DIR}:")
    for f in sorted(files):
        print(f"    • {f.name} ({round(f.stat().st_size / 1024, 1)} KB)")

def cmd_verify():
    master_file = OUTPUT_DIR / "tamil_1min_indic_parler_master.wav"
    if not master_file.exists():
        wavs = list(OUTPUT_DIR.glob("*.wav"))
        if not wavs:
            print(f"[-] Error: No WAV files found in {OUTPUT_DIR}. Run 'download' first.")
            sys.exit(1)
        master_file = wavs[0]

    print("\n" + "=" * 60)
    print(" AI4BHARAT INDIC PARLER-TTS TAMIL AUDIO REPORT")
    print("=" * 60)
    print(f"File: {master_file}")
    
    with wave.open(str(master_file), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        duration_sec = n_frames / float(framerate)
        
        frames = wf.readframes(n_frames)
        fmt = f"<{n_frames * n_channels}h"
        samples = struct.unpack(fmt, frames)
        
        sum_sq = sum(s ** 2 for s in samples)
        rms = math.sqrt(sum_sq / len(samples))
        peak = max(abs(s) for s in samples)

    print(f"Duration:     {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")
    print(f"Sample Rate:  {framerate} Hz")
    print(f"Channels:     {n_channels} (mono)")
    print(f"Bit Depth:    {sampwidth * 8}-bit PCM")
    print(f"RMS Energy:   {rms:.1f} / 32768 ({round(rms / 32768 * 100, 1)}% full-scale)")
    print(f"Peak Energy:  {peak} / 32768 ({round(peak / 32768 * 100, 1)}% full-scale)")
    print("=" * 60)
    
    summary_file = OUTPUT_DIR / "generation_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print("Synthesis Details:")
            print(f"  Model:        {data.get('model')}")
            print(f"  Language:     {data.get('language')}")
            print(f"  Institution:  {data.get('institution')}")
            if "speaker_jaya" in data:
                print(f"  Speaker Jaya:    {data['speaker_jaya'].get('duration_sec')}s")
            if "speaker_kavitha" in data:
                print(f"  Speaker Kavitha: {data['speaker_kavitha'].get('duration_sec')}s")
            print("=" * 60)
        except Exception:
            pass

    if duration_sec >= 45.0:
        print("[+] VERIFICATION PASSED: Full ~1-minute Tamil continuous narrative generated successfully!")
    else:
        print(f"[!] Warning: Duration ({duration_sec:.1f}s) is less than 45s.")

def cmd_run():
    print("=" * 70)
    print(" AI4BHARAT INDIC PARLER-TTS: END-TO-END TAMIL TTS PIPELINE")
    print("=" * 70)
    cmd_push()
    print("\n[*] Monitoring Kaggle GPU job execution...")
    start_time = time.time()
    while True:
        status = cmd_status()
        if "KernelWorkerStatus.COMPLETE" in status:
            print("\n[+] Kaggle GPU execution finished successfully!")
            break
        elif "KernelWorkerStatus.ERROR" in status or "KernelWorkerStatus.CANCEL" in status:
            print(f"\n[-] Execution failed with status: {status}")
            cmd_logs()
            sys.exit(1)
        elapsed = int(time.time() - start_time)
        print(f"    [{elapsed}s elapsed] Waiting for completion... (checks every 20s)")
        time.sleep(20)
    print("\n[*] Retrieving generated audio assets...")
    cmd_download()
    cmd_verify()

def main():
    parser = argparse.ArgumentParser(description="Indic Parler-TTS CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("push", help="Push kernel to Kaggle")
    subparsers.add_parser("status", help="Check Kaggle kernel status")
    subparsers.add_parser("logs", help="View Kaggle kernel logs")
    subparsers.add_parser("download", help="Download output files")
    subparsers.add_parser("verify", help="Inspect and verify downloaded audio")
    subparsers.add_parser("run", help="Run end-to-end pipeline")
    args = parser.parse_args()
    if args.command == "push":
        cmd_push()
    elif args.command == "status":
        cmd_status()
    elif args.command == "logs":
        cmd_logs()
    elif args.command == "download":
        cmd_download()
    elif args.command == "verify":
        cmd_verify()
    elif args.command == "run":
        cmd_run()

if __name__ == "__main__":
    main()
