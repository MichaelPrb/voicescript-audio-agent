import subprocess
import json
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

def extract_metadata(file_path):
    """Extracts basic audio metadata using ffprobe."""
    print(f"[INFO] Extracting metadata: {file_path}")
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', file_path]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(result.stdout)
        
        audio_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None)
        format_info = data.get('format', {})
        duration = float(format_info.get('duration', 0))
        
        return {
            "file_name": os.path.basename(file_path),
            "duration_seconds": round(duration, 2),
            "bitrate": int(format_info.get('bit_rate', 0)),
            "sample_rate": int(audio_stream.get('sample_rate', 0)) if audio_stream else None,
            "channels": int(audio_stream.get('channels', 0)) if audio_stream else None
        }
    except Exception as e:
        print(f"[ERROR] Failed to extract metadata for {file_path}: {e}")
        return None

def analyze_audio_quality(file_path):
    """Analyzes audio for silence and clipping using ffmpeg filters."""
    print(f"[INFO] Analyzing signal quality: {file_path}")
    cmd = ['ffmpeg', '-i', file_path, '-af', 'silencedetect=noise=-40dB:d=2,volumedetect', '-f', 'null', '-']
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stderr_output = result.stderr
        
        mean_vol_match = re.search(r'mean_volume:\s+([\-\d\.]+)\s+dB', stderr_output)
        max_vol_match = re.search(r'max_volume:\s+([\-\d\.]+)\s+dB', stderr_output)
        
        mean_vol = float(mean_vol_match.group(1)) if mean_vol_match else -99.0
        max_vol = float(max_vol_match.group(1)) if max_vol_match else -99.0
        
        silence_durations = re.findall(r'silence_duration:\s+([\d\.]+)', stderr_output)
        total_silence = sum(float(d) for d in silence_durations)
        
        return {
            "avg_volume_db": mean_vol,
            "clipping_detected": max_vol >= 0.0,
            "total_silence_seconds": round(total_silence, 2)
        }
    except Exception as e:
        print(f"[ERROR] Failed to analyze quality for {file_path}: {e}")
        return None

def get_llm_insights(audio_quality_data, provider="gemini"):
    """Generates actionable QA insights using an LLM Adapter."""
    print(f"[INFO] Requesting insights via {provider.upper()} provider...")
    
    prompt = f"""
    You are an expert Audio Engineer for a legal transcription service.
    Analyze the following audio quality metrics:
    {json.dumps(audio_quality_data, indent=2)}
    
    Provide actionable insights regarding the audio usability, specifically focusing on clipping, volume, and silence ratio.
    Respond ONLY with a valid JSON array of strings (max 2 sentences per string).
    Example: ["This audio contains extended silence. Recommend trimming.", "Clipping detected. Verify microphone levels."]
    """
    
    try:
        if provider == "gemini":
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel('gemini-3.5-flash')
            response = model.generate_content(prompt)
            
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
            
        elif provider == "mock":
            return ["System functioning. Audio requires manual review due to mock mode."]
            
    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        return ["Error generating automated insights."]

def process_pipeline(file_path, llm_provider="gemini"):
    """Executes the full extraction, analysis, and generative QA pipeline."""
    metadata = extract_metadata(file_path)
    if not metadata: 
        return None
        
    quality = analyze_audio_quality(file_path)
    if not quality:
        return None
    
    # Calculate Silence Ratio
    if metadata["duration_seconds"] > 0:
        silence_ratio = quality["total_silence_seconds"] / metadata["duration_seconds"]
    else:
        silence_ratio = 0
        
    quality["silence_ratio"] = round(silence_ratio, 3)
    del quality["total_silence_seconds"]
    
    # Fetch generative insights
    issues = get_llm_insights(quality, provider=llm_provider)
    
    # Assemble final output structure
    final_report = metadata
    final_report["audio_quality"] = quality
    final_report["issues"] = issues
    
    return final_report

if __name__ == "__main__":
    files_to_test = ["audio/bad_audio.mp3", "audio/moonlight-plaza.mp3"]
    final_outputs = []
    
    print("=== AUDIO INTELLIGENCE PIPELINE ===")
    for file in files_to_test:
        if os.path.exists(file):
            report = process_pipeline(file, llm_provider="gemini")
            if report:
                final_outputs.append(report)
            print("-" * 50)
        else:
            print(f"[WARN] File not found: {file}")
            
    print("\n=== STRUCTURED OUTPUT ===")
    print(json.dumps(final_outputs, indent=4))