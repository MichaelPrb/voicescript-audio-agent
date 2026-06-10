# Audio Intelligence Agent: Analysis & QA Pipeline

## Overview
This repository contains a robust, scalable audio analysis pipeline designed to process legal deposition recordings. It extracts low-level signal metadata via `ffmpeg` and utilizes an LLM layer to generate structured, human-readable Quality Assurance (QA) reports.

## Architecture & Design Decisions

### 1. Signal Processing via Subprocess
**Decision:** Utilized `ffmpeg` and `ffprobe` via Python's `subprocess` module to parse audio streams.
**Rationale:** `ffmpeg` processes media at the C-binary level, making it exponentially faster and more memory-efficient when handling hours of deposition audio compared to heavy Python audio libraries.

### 2. The LLM Adapter Pattern (Preventing Vendor Lock-in)
**Decision:** Implemented an Adapter Pattern for the LLM generative layer. 
**Rationale:** Real-world AI systems should not be tightly coupled to a single vendor. The pipeline currently uses the `gemini-3.5-flash` model for rapid, cost-effective evaluation, but the architecture allows for a seamless switch to OpenAI or local mock generation by simply changing the `provider` parameter.

### 3. Structured Output Enforcement
**Decision:** The final output is strictly coerced into a predefined JSON schema.
**Rationale:** For integration with downstream full-stack systems and front-end dashboards, deterministic and predictable data structures are non-negotiable.

## Pipeline Workflow
1.  **Metadata Extraction:** `ffprobe` extracts duration, bitrate, sample rate, and channel configurations.
2.  **Quality Analysis:** `ffmpeg` applies the `silencedetect` and `volumedetect` filters to identify extended quiet periods (-40dB) and potential peak clipping (≥0.0dB).
3.  **Data Aggregation:** Python Regex processes the `stderr` streams, computing the silence ratio.
4.  **Generative Insight:** The aggregated data is passed to the LLM Adapter, returning structured, actionable insights regarding audio usability for legal transcription.

## Prerequisites & Installation
* Python 3.8+
* `ffmpeg` installed and accessible in the system PATH.
* Install requirements: 
  ```bash
  pip install google-generativeai python-dotenv

## How to Run
1.  Clone the repository and place the target audio files in the `audio/ directory`.
2.  Create a `.env` file in the root directory and add your API Key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
3.  Execute the main pipeline:
```bash
python main.py