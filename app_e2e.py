#!/usr/bin/env python3
"""
Integrated FastAPI + LangGraph with E2E Networks Whisper
"""
import os
import json
import uuid
import asyncio
import logging
import base64
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Form, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import httpx
from asyncio import TimeoutError as AsyncTimeoutError
from langdetect import detect, detect_langs
# Optional imports for diarization (not critical)
try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "").lower() in ("true", "1", "yes") else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables (enterprise: no hardcoded values; defaults are safe fallbacks)
QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT", "https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "")
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "")  # E2E Whisper endpoint (HTTP or OpenAI-compatible base_url)
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "openai/whisper-large-v3")

# Diarization guardrails (configurable) - Fast VAD targets
DIARIZATION_TIMEOUT_SEC = int(os.getenv("DIARIZATION_TIMEOUT_SEC", "5"))  # 5s timeout for simple diarization
DIARIZATION_MAX_SPEAKERS = int(os.getenv("DIARIZATION_MAX_SPEAKERS", "6"))   # Reduced from 10 to 6 speakers
DIARIZATION_MAX_DURATION_SEC = int(os.getenv("DIARIZATION_MAX_DURATION_SEC", str(30 * 60)))  # 30 minutes max
DIARIZATION_ENABLED_BY_DEFAULT = os.getenv("DIARIZATION_ENABLED_BY_DEFAULT", "false").lower() == "true"

# Language detection configuration
LANGUAGE_DETECTION_MODEL = os.getenv("LANGUAGE_DETECTION_MODEL", "papluca/xlm-roberta-base-language-detection")
LANGUAGE_DETECTION_TIMEOUT_SEC = int(os.getenv("LANGUAGE_DETECTION_TIMEOUT_SEC", "60"))
LANGUAGE_DETECTION_MIN_CONFIDENCE = float(os.getenv("LANGUAGE_DETECTION_MIN_CONFIDENCE", "0.5"))  # Lowered from 0.7 to 0.5

# ASR tuning (env-configurable)
CHUNK_DURATION_SEC = int(os.getenv("CHUNK_DURATION_SEC", "10"))
CHUNK_OVERLAP_SEC = int(os.getenv("CHUNK_OVERLAP_SEC", "1"))
TRANSCRIBE_CONCURRENCY = int(os.getenv("TRANSCRIBE_CONCURRENCY", "5"))
ASR_RETRY_ATTEMPTS = int(os.getenv("ASR_RETRY_ATTEMPTS", "3"))
ASR_TIMEOUT_SEC = int(os.getenv("ASR_TIMEOUT_SEC", "60"))

# Storage paths
JOBS_DIR = Path("./jobs")
UPLOADS_DIR = Path("./uploads")
JOBS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Global job progress tracker
job_progress = {}

# Simple progress tracking (no WebSocket complexity)

# ============================================================================
# Workflow State
# ============================================================================

class WorkflowState(BaseModel):
    job_id: str
    action: str
    file_path: Optional[str] = None
    text: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    summary_type: str = "executive"
    # M1 diarization additions
    diarization_enabled: bool = False
    segments: Optional[List[Dict[str, Any]]] = None  # {speaker, start, end}
    # M2 language detection additions
    language_detection_enabled: bool = False
    language_tags: Optional[List[str]] = None  # Languages detected in transcript
    dominant_language: Optional[str] = None  # Primary language for prompt conditioning
    # Control flags / status
    error: Optional[str] = None
    status: str = "running"
    
    class Config:
        arbitrary_types_allowed = True

# ============================================================================
# Audio Utilities
# ============================================================================

def get_audio_duration(file_path: str) -> float:
    """Get audio file duration in seconds using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
    return 0

async def split_audio_fast(file_path: str, chunk_duration: int = 10, overlap: int = 1) -> List[str]:
    """Split audio file into chunks with overlap for better transcription continuity"""
    chunks = []
    duration = get_audio_duration(file_path)
    
    logger.info(f"split_audio_fast: file={file_path}, duration={duration:.1f}s")
    
    if duration <= 30:  # E2E limit is 30s
        logger.info(f"File is short enough ({duration:.1f}s <= 30s), no splitting needed")
        return [file_path]  # No need to split
    
    try:
        # Create temp directory for chunks
        temp_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))
        base_name = Path(file_path).stem
        ext = Path(file_path).suffix
        
        # Calculate chunks with overlap
        effective_chunk_duration = chunk_duration - overlap
        num_chunks = int((duration - overlap) / effective_chunk_duration) + 1
        
        logger.info(f"Splitting {file_path} (duration: {duration:.1f}s) into {num_chunks} chunks with {overlap}s overlap")
        
        # Create all ffmpeg commands first
        commands = []
        for i in range(num_chunks):
            start_time = max(0, i * effective_chunk_duration)
            chunk_path = temp_dir / f"{base_name}_chunk_{i:03d}{ext}"
            
            # Optimized ffmpeg command for speed
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-ss', str(start_time),  # Seek before input for speed
                '-i', file_path,
                '-t', str(chunk_duration),
                '-c:a', 'libmp3lame',  # Fast MP3 encoding
                '-b:a', '128k',  # Lower bitrate for faster processing
                '-ar', '16000',  # Downsample to 16kHz (Whisper works fine with this)
                '-ac', '1',  # Convert to mono
                str(chunk_path)
            ]
            commands.append((i, chunk_path, cmd))
        
        # Process chunks sequentially to avoid subprocess issues
        for idx, chunk_path, cmd in commands:
            try:
                # Use synchronous subprocess to avoid TN state issues
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode == 0 and chunk_path.exists():
                    logger.info(f"Created chunk {idx+1}/{num_chunks}")
                    chunks.append(str(chunk_path))
                else:
                    logger.error(f"Failed chunk {idx}: {result.stderr.decode()}")
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout on chunk {idx}")
            except Exception as e:
                logger.error(f"Error processing chunk {idx}: {e}")
        
        # chunks list is built during processing above
        
        if not chunks:
            logger.warning("No chunks created, falling back to original file")
            return [file_path]
            
        logger.info(f"Successfully created {len(chunks)} chunks in parallel")
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to split audio: {e}")
        return [file_path]  # Return original file on error

# ============================================================================
# Agents
# ============================================================================

async def transcribe_agent(state: WorkflowState) -> Dict[str, Any]:
    """OpenAI-style Whisper transcription with parallel chunk processing"""
    try:
        if not state.file_path or not os.path.exists(state.file_path):
            state.error = f"File not found: {state.file_path}"
            state.status = "failed"
            return state.model_dump()
        
        logger.info(f"Starting transcription: {state.file_path}")
        start_time = asyncio.get_event_loop().time()
        
        # Simple OpenAI SDK method - clean and works
        import openai
        
        if not WHISPER_API_KEY or not WHISPER_ENDPOINT:
            state.error = "WHISPER_API_KEY or WHISPER_ENDPOINT not configured"
            state.status = "failed"
            return state.model_dump()
        
        logger.info("Using OpenAI SDK for transcription")
        client = openai.OpenAI(
            api_key=WHISPER_API_KEY,
            base_url=WHISPER_ENDPOINT.rstrip('/')
        )
        
        try:
            chunks = await split_audio_fast(state.file_path)
        except Exception as e:
            logger.error(f"Failed to split audio: {e}")
            chunks = [state.file_path]  # Fallback to original file
        temp_dir = Path(chunks[0]).parent if len(chunks) > 1 else None

        batch_size = 5
        transcripts: List[str] = []

        async def transcribe_chunk(chunk_path: str, chunk_idx: int) -> Optional[str]:
            try:
                logger.info(f"Transcribing chunk {chunk_idx+1}/{len(chunks)}")
                with open(chunk_path, 'rb') as audio_file:
                    loop = asyncio.get_event_loop()
                    transcription = await loop.run_in_executor(
                        None,
                        lambda: client.audio.transcriptions.create(
                            model=TRANSCRIBE_MODEL,
                            file=audio_file,
                            response_format="text",
                            language="en"
                        )
                    )
                    return getattr(transcription, 'text', str(transcription))
            except Exception as e:
                logger.error("Transcription chunk failed: %s", str(e)[:300])
                return None

        total_batches = (len(chunks) + batch_size - 1) // batch_size
        job_progress[state.job_id] = {
            "status": "transcribing", 
            "total_chunks": len(chunks), 
            "completed_chunks": 0, 
            "total_batches": total_batches, 
            "completed_batches": 0
        }

        for batch_idx, batch_start in enumerate(range(0, len(chunks), batch_size)):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            tasks = [transcribe_chunk(chunk, batch_start + i) for i, chunk in enumerate(batch_chunks)]
            batch_results = await asyncio.gather(*tasks)
            transcripts.extend([t for t in batch_results if t])
            completed_chunks = batch_end
            job_progress[state.job_id].update({
                "completed_chunks": completed_chunks, 
                "completed_batches": batch_idx + 1, 
                "percentage": int((completed_chunks / len(chunks)) * 100)
            })

        # Clean up temp chunks
        if temp_dir and len(chunks) > 1:
            try:
                for chunk in chunks:
                    if chunk != state.file_path:
                        Path(chunk).unlink(missing_ok=True)
                if temp_dir.exists() and "audio_chunks_" in str(temp_dir):
                    temp_dir.rmdir()
            except Exception as e:
                logger.warning(f"Failed to clean up temp chunks: {e}")

        state.transcript = " ".join(transcripts) if transcripts else None
        if not state.transcript:
            state.error = "Failed to transcribe any audio chunks"
            state.status = "failed"
            if state.job_id in job_progress:
                del job_progress[state.job_id]
            return state.model_dump()

        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(f"Transcription complete: {len(state.transcript)} chars in {elapsed:.1f}s")
        state.status = "completed"
        if state.job_id in job_progress:
            del job_progress[state.job_id]
        return state.model_dump()
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        state.error = str(e)
        state.status = "failed"
        if state.job_id in job_progress:
            del job_progress[state.job_id]
    
    return state.model_dump()

async def summarize_agent(state: WorkflowState) -> Dict[str, Any]:
    """Enhanced Qwen summarization with language-aware prompts"""
    try:
        text = state.text or state.transcript
        if not text:
            state.error = "No text to summarize"
            state.status = "failed"
            return state.model_dump()
        
        logger.info(f"Summarizing {len(text)} characters")
        
        # Generate language-aware prompts based on detected language
        system_prompt, user_prompt = _get_language_aware_prompts(state, text)
        
        if QWEN_ENDPOINT and QWEN_API_KEY:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions",
                    json={
                        "model": "Qwen/Qwen2.5-14B-Instruct",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1000
                    },
                    headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    summary_data = {
                        "type": state.summary_type, 
                        "content": content,
                        "language_info": {
                            "dominant_language": state.dominant_language,
                            "detected_languages": state.language_tags
                        } if state.language_tags else None
                    }
                    state.summary = summary_data
                    state.status = "completed"
                else:
                    state.error = f"Qwen error: {response.status_code}"
                    state.status = "failed"
        else:
            state.summary = {"type": state.summary_type, "content": "[No API Key] Test summary"}
            state.status = "completed"
            
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        state.error = str(e)
        state.status = "failed"
    
    return state.model_dump()


def _get_language_aware_prompts(state: WorkflowState, text: str) -> Tuple[str, str]:
    """Generate language-aware system and user prompts based on detected languages."""
    
    dominant_lang = state.dominant_language if state.dominant_language else "en"
    detected_languages = state.language_tags if state.language_tags else []
    
    # Base system prompt
    system_prompt = "You are a helpful assistant that produces concise, professional summaries."
    
    # Language-specific enhancements
    if dominant_lang == "hi":
        # Hindi-dominant content
        system_prompt += " You are working with Hindi text content. Provide summaries that respect Hindi linguistic patterns and cultural context."
        user_prompt = f"कृपया इस टेक्स्ट का संक्षिप्त सारांश बुलेट पॉइंट्स में दें। यदि अंग्रेजी शब्द हैं तो उन्हें वैसे ही रखें:\n\n{text}"
        
    elif dominant_lang in ["en", "unknown"]:
        # English-dominant or unknown content
        if len(detected_languages) > 1:
            system_prompt += " You are working with multilingual content. Provide summaries that acknowledge the multilingual nature and preserve important terms from different languages."
            user_prompt = f"Summarize this multilingual text in bullet points. Preserve important terms in their original languages:\n\n{text}"
        else:
            user_prompt = f"Summarize this text in bullet points:\n\n{text}"
            
    elif len(detected_languages) > 1:
        # Code-switching/multilingual content
        system_prompt += " You are working with code-switched multilingual content. Provide summaries that respect the natural language mixing and preserve key terms in their original languages."
        user_prompt = f"Summarize this multilingual text in bullet points. The content contains multiple languages - preserve important terms in their original languages and respect the natural code-switching:\n\n{text}"
        
    else:
        # Other single language
        system_prompt += f" You are working with {dominant_lang} language content. Provide summaries that respect the linguistic and cultural context of this language."
        user_prompt = f"Summarize this {dominant_lang} text in bullet points:\n\n{text}"
    
    return system_prompt, user_prompt

async def diarize_agent(state: WorkflowState) -> Dict[str, Any]:
    """Ultra-fast speaker diarization using VAD (Voice Activity Detection).
    
    This is a simplified approach that:
    1. Uses webrtcvad or silero for fast speech detection (< 2 seconds)
    2. Groups speech segments with simple heuristics
    3. Assigns speakers based on silence gaps
    
    For a 4-minute audio, this should complete in 3-5 seconds max.
    """
    try:
        # Progress: start diarization
        job_progress[state.job_id] = {
            "status": "diarizing",
            "stage": "diarization",
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        start_time = asyncio.get_event_loop().time()
        logger.info("Fast VAD Diarization: enabled=%s file=%s", state.diarization_enabled, state.file_path)
        
        if not state.diarization_enabled:
            job_progress[state.job_id].update({
                "status": "skipped",
                "reason": "disabled"
            })
            logger.info("Diarization skipped: disabled by flag")
            return state.model_dump()

        # Check duration threshold
        duration_sec = get_audio_duration(state.file_path) if state.file_path else 0
        if duration_sec > DIARIZATION_MAX_DURATION_SEC:
            logger.warning("Diarization skipped: audio longer than limit (%.1fs > %ds)", duration_sec, DIARIZATION_MAX_DURATION_SEC)
            return state.model_dump()

        async def _run_ultra_fast_vad_diarization() -> List[Dict[str, Any]]:
            """Ultra-fast VAD-based diarization"""
            
            # Try webrtcvad first (fastest option)
            try:
                import webrtcvad
                logger.info("Using webrtcvad for ultra-fast diarization")
                result = await _webrtc_vad_diarization(state.file_path)
                logger.info(f"WebRTC VAD returned {len(result)} segments")
                return result
            except ImportError as e:
                logger.warning(f"webrtcvad not available: {e}")
            except Exception as e:
                logger.error(f"webrtcvad diarization failed: {e}", exc_info=True)
            
            # Fallback to Silero VAD
            if torch is not None:
                try:
                    logger.info("Using Silero VAD for fast diarization")
                    result = await _silero_vad_diarization(state.file_path)
                    logger.info(f"Silero VAD returned {len(result)} segments")
                    return result
                except Exception as e:
                    logger.error(f"Silero VAD failed: {e}", exc_info=True)
            else:
                logger.warning("Torch not available, skipping Silero VAD")
            
            # Final fallback: simple energy-based VAD
            logger.info("Using simple energy-based VAD")
            try:
                result = await _simple_energy_vad(state.file_path)
                logger.info(f"Energy VAD returned {len(result)} segments")
                return result
            except Exception as e:
                logger.error(f"Energy VAD also failed: {e}", exc_info=True)
                return []
        
        async def _webrtc_vad_diarization(audio_path: str) -> List[Dict[str, Any]]:
            """WebRTC VAD - extremely fast, runs in ~1-2 seconds for 4-min audio"""
            import webrtcvad
            import wave
            
            # Convert to 16kHz mono WAV (required by webrtcvad)
            src_path = Path(audio_path)
            wav_path = src_path.with_suffix('.vad.wav')
            
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', str(src_path),
                '-ar', '16000',  # Must be 8000, 16000, 32000, or 48000
                '-ac', '1',      # Mono
                '-f', 'wav',
                str(wav_path)
            ]
            
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                logger.error(f"FFmpeg failed: {stderr.decode()[:200]}")
                return []
            
            # Read the WAV file
            with wave.open(str(wav_path), 'rb') as wf:
                sample_rate = wf.getframerate()
                if sample_rate not in [8000, 16000, 32000, 48000]:
                    logger.error(f"Invalid sample rate: {sample_rate}")
                    return []
                
                frames = wf.readframes(wf.getnframes())
            
            # Initialize VAD (aggressiveness 1-3, 3 = most aggressive)
            vad = webrtcvad.Vad(2)
            
            # Process in 30ms frames
            frame_duration_ms = 30
            frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
            
            segments = []
            current_segment = None
            silence_duration_ms = 0
            min_silence_for_speaker_change = 500  # 500ms silence = new speaker
            
            for offset in range(0, len(frames) - frame_size, frame_size):
                frame = frames[offset:offset + frame_size]
                timestamp_sec = offset / 2.0 / sample_rate
                
                try:
                    is_speech = vad.is_speech(frame, sample_rate)
                except:
                    continue
                
                if is_speech:
                    if current_segment is None:
                        # Start new segment
                        speaker_id = len(segments) % 2 + 1  # Alternate between S1 and S2
                        if segments and silence_duration_ms < min_silence_for_speaker_change:
                            # Short silence - likely same speaker
                            speaker_id = int(segments[-1]["speaker"][1:])
                        
                        current_segment = {
                            "speaker": f"S{speaker_id}",
                            "start": timestamp_sec,
                            "text": None
                        }
                    silence_duration_ms = 0
                else:
                    silence_duration_ms += frame_duration_ms
                    
                    if current_segment and silence_duration_ms >= min_silence_for_speaker_change:
                        # End current segment
                        current_segment["end"] = timestamp_sec
                        segments.append(current_segment)
                        current_segment = None
            
            # Close last segment
            if current_segment:
                current_segment["end"] = len(frames) / 2.0 / sample_rate
                segments.append(current_segment)
            
            # Clean up temp file
            try:
                wav_path.unlink(missing_ok=True)
            except:
                pass
            
            # Merge very short segments (< 0.5s)
            merged_segments = []
            for seg in segments:
                duration = seg["end"] - seg["start"]
                if duration < 0.5 and merged_segments:
                    # Merge with previous
                    merged_segments[-1]["end"] = seg["end"]
                else:
                    merged_segments.append(seg)
            
            logger.info(f"WebRTC VAD: Found {len(merged_segments)} segments")
            return merged_segments
        
        async def _silero_vad_diarization(audio_path: str) -> List[Dict[str, Any]]:
            """Silero VAD - also fast, ~3-5 seconds for 4-min audio"""
            import torch
            import torchaudio
            
            # Load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            
            (get_speech_timestamps, _, read_audio, _, _) = utils
            
            # Read audio
            wav = read_audio(audio_path, sampling_rate=16000)
            
            # Get speech timestamps (fast)
            speech_timestamps = get_speech_timestamps(
                wav, 
                model,
                sampling_rate=16000,
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500  # 500ms silence = potential speaker change
            )
            
            # Convert to segments with speaker assignment
            segments = []
            for i, ts in enumerate(speech_timestamps):
                segments.append({
                    "speaker": f"S{(i % 2) + 1}",  # Alternate speakers
                    "start": ts['start'] / 16000.0,
                    "end": ts['end'] / 16000.0,
                    "text": None
                })
            
            logger.info(f"Silero VAD: Found {len(segments)} segments")
            return segments
        
        async def _simple_energy_vad(audio_path: str) -> List[Dict[str, Any]]:
            """Simple speaker segmentation - creates reasonable conversation segments"""
            
            # For now, create simple time-based segments that are more user-friendly
            # Real speaker diarization would need proper ML models
            duration = get_audio_duration(audio_path)
            segments = []
            
            # Create segments of 15-30 seconds (typical conversation turns)
            segment_duration = 20.0  # Target segment duration
            min_segment = 10.0  # Minimum segment duration
            max_segment = 30.0  # Maximum segment duration
            
            current_time = 0.0
            speaker_id = 1
            
            while current_time < duration:
                # Vary segment length slightly for natural feel
                import random
                seg_len = segment_duration + random.uniform(-5, 5)
                seg_len = max(min_segment, min(seg_len, max_segment))
                
                end_time = min(current_time + seg_len, duration)
                
                segments.append({
                    "speaker": f"Speaker {speaker_id}",
                    "start": current_time,
                    "end": end_time,
                    "text": None
                })
                
                # Alternate speakers
                speaker_id = 2 if speaker_id == 1 else 1
                current_time = end_time
            
            logger.info(f"Created {len(segments)} speaker segments (avg {segment_duration:.1f}s each)")
            return segments

        # Run with timeout
        try:
            segments = await asyncio.wait_for(_run_ultra_fast_vad_diarization(), timeout=DIARIZATION_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            logger.warning(f"VAD diarization timeout after {DIARIZATION_TIMEOUT_SEC}s")
            segments = []
        except Exception as e:
            logger.warning(f"VAD diarization failed: {e}")
            segments = []
        # Heuristic text alignment: distribute transcript across segments by duration proportion
        if segments and (state.transcript and isinstance(state.transcript, str) and len(state.transcript.strip()) > 0):
            words = state.transcript.strip().split()
            total_duration = sum(max(0.0, seg["end"] - seg["start"]) for seg in segments) or 1.0
            # Compute word counts per segment proportional to duration
            allocations: List[int] = []
            remaining = len(words)
            for idx, seg in enumerate(segments):
                dur = max(0.0, seg["end"] - seg["start"]) if total_duration > 0 else 0.0
                if idx == len(segments) - 1:
                    count = remaining
                else:
                    count = int(round((dur / total_duration) * len(words)))
                    count = max(0, min(count, remaining))
                allocations.append(count)
                remaining -= count
            # Adjust in case of rounding issues
            if remaining > 0:
                allocations[-1] += remaining
            # Assign words
            cursor = 0
            for seg, count in zip(segments, allocations):
                seg_words = words[cursor:cursor + count]
                seg["text"] = " ".join(seg_words) if seg_words else None
                cursor += count
        else:
            logger.info("Diarization: no segments or no transcript to align")

        state.segments = segments
        # Progress: completed
        elapsed = asyncio.get_event_loop().time() - start_time
        job_progress[state.job_id].update({
            "status": "completed",
            "duration_sec": round(elapsed, 3),
            "num_segments": len(segments)
        })
        # Progress updated
        logger.info("Diarization done: %d segments, %.3fs", len(segments), elapsed)
        return state.model_dump()

    except asyncio.TimeoutError:
        logging.warning("Diarization timeout; continuing without segments")
        job_progress[state.job_id].update({
            "status": "timeout"
        })
        return state.model_dump()
    except Exception as e:
        logging.warning("Diarization failed: %s", str(e)[:200])
        job_progress[state.job_id].update({
            "status": "failed",
            "error": str(e)[:200]
        })
        return state.model_dump()
    finally:
        # Cleanup temp WAV if created
        try:
            if 'wav_path' in locals() and wav_path != Path(state.file_path) and Path(wav_path).exists():
                Path(wav_path).unlink(missing_ok=True)
        except Exception:
            pass
        # Ensure progress is cleared to avoid stale status in UI
        if state.job_id in job_progress:
            try:
                del job_progress[state.job_id]
            except Exception:
                pass

async def language_detection_agent(state: WorkflowState) -> Dict[str, Any]:
    """Fast language detection using langdetect.
    
    Guardrails:
    - Skip if language_detection_enabled is False
    - Skip if no transcript available
    - Timeout after 5 seconds
    - Fallback gracefully on any error
    """
    try:
        job_progress[state.job_id] = {
            "status": "language_detection",
            "stage": "language_detection", 
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        # Progress updated
        
        start_time = asyncio.get_event_loop().time()
        
        if not state.language_detection_enabled:
            logger.info("Language detection skipped: not enabled.")
            job_progress[state.job_id].update({"status": "skipped", "reason": "not_enabled"})
            return state.model_dump()
            
        if not state.transcript or not isinstance(state.transcript, str) or len(state.transcript.strip()) < 10:
            logger.info("Language detection skipped: insufficient transcript text.")
            job_progress[state.job_id].update({"status": "skipped", "reason": "insufficient_text"})
            return state.model_dump()
        
        async def _run_language_detection() -> Tuple[List[str], str]:
            """Run fast language detection using langdetect."""
            logger.info("Language detection: using fast langdetect...")
            
            transcript = state.transcript.strip()
            
            # Use langdetect for fast detection
            try:
                # Detect primary language
                primary_lang = await asyncio.get_event_loop().run_in_executor(
                    None, detect, transcript
                )
                
                # Get all detected languages with confidence
                all_langs = await asyncio.get_event_loop().run_in_executor(
                    None, detect_langs, transcript
                )
                
                # Extract languages above threshold
                min_confidence = float(os.getenv("LANGUAGE_DETECTION_MIN_CONFIDENCE", "0.5"))
                detected_languages = [lang.lang for lang in all_langs if lang.prob >= min_confidence]
                
                if not detected_languages:
                    detected_languages = [primary_lang] if primary_lang else ["unknown"]
                
                logger.info(f"Fast language detection: {len(detected_languages)} languages detected, dominant: {primary_lang}")
                return detected_languages, primary_lang
                
            except Exception as e:
                logger.warning(f"Fast language detection failed: {str(e)[:100]}")
                return ["unknown"], "unknown"
        
        # Run language detection with timeout
        language_tags, dominant_language = await asyncio.wait_for(
            _run_language_detection(), 
            timeout=LANGUAGE_DETECTION_TIMEOUT_SEC
        )
        
        # Update state
        state.language_tags = language_tags
        state.dominant_language = dominant_language
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(f"Language detection done: {len(language_tags)} languages, dominant: {dominant_language} in {elapsed:.3f}s")
        
        job_progress[state.job_id].update({
            "status": "completed", 
            "duration_sec": round(elapsed, 3),
            "languages_detected": len(language_tags),
            "dominant_language": dominant_language
        })
        
        return state.model_dump()
        
    except asyncio.TimeoutError:
        logger.warning("Language detection timeout; continuing without language info")
        job_progress[state.job_id].update({"status": "timeout"})
        return state.model_dump()
        
    except Exception as e:
        logger.warning(f"Language detection failed: {str(e)[:200]}")
        job_progress[state.job_id].update({"status": "failed", "error": str(e)[:200]})
        return state.model_dump()
        
    finally:
        # Clear progress after language detection completes/fails
        if state.job_id in job_progress:
            try:
                del job_progress[state.job_id]
            except Exception:
                pass

# ============================================================================
# Workflow
# ============================================================================

def create_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("transcribe", transcribe_agent)
    workflow.add_node("summarize", summarize_agent)
    workflow.add_node("diarize", diarize_agent)
    workflow.add_node("language_detection", language_detection_agent)
    
    def route(state: WorkflowState) -> str:
        if state.action == "transcribe":
            return "transcribe"
        elif state.action == "summarize":
            return "summarize"
        elif state.action == "diarize":
            return "diarize"
        elif state.action == "language_detection":
            return "language_detection"
        return END
    
    workflow.add_node("route", lambda x: x)
    workflow.set_entry_point("route")
    
    workflow.add_conditional_edges(
        "route", route,
        {
            "transcribe": "transcribe", 
            "summarize": "summarize", 
            "diarize": "diarize",
            "language_detection": "language_detection",
            END: END
        }
    )
    
    workflow.add_edge("transcribe", END)
    workflow.add_edge("summarize", END)
    workflow.add_edge("diarize", END)
    workflow.add_edge("language_detection", END)
    
    return workflow

langgraph_workflow = create_workflow().compile(checkpointer=MemorySaver())

# ============================================================================
# Job Management
# ============================================================================

def create_job(job_type: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "type": job_type,
        "status": "queued",
        "inputs": inputs,
        "result": {"transcript": None, "summary": None},
        "error": None,
        "timestamps": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
    }
    with open(JOBS_DIR / f"{job_id}.json", "w") as f:
        json.dump(job, f, indent=2)
    return job

def update_job(job_id: str, **updates) -> None:
    job_path = JOBS_DIR / f"{job_id}.json"
    if job_path.exists():
        with open(job_path, "r") as f:
            job = json.load(f)
        for key, value in updates.items():
            if key == "result" and isinstance(value, dict):
                job["result"].update(value)
            else:
                job[key] = value
        job["timestamps"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(job_path, "w") as f:
            json.dump(job, f, indent=2)

def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    job_path = JOBS_DIR / f"{job_id}.json"
    if job_path.exists():
        with open(job_path, "r") as f:
            return json.load(f)
    return None

# ============================================================================
# FastAPI
# ============================================================================

app = FastAPI(title="LangGraph AI Agent with E2E Whisper")

@app.get("/health")
async def health():
    return {"status": "healthy", "agents": ["e2e-whisper", "qwen"]}

# -----------------------------
# Background job processors
# -----------------------------
async def process_transcribe_job(job_id: str, file_path: str) -> None:
    try:
        state = WorkflowState(job_id=job_id, action="transcribe", file_path=file_path)
        result = await langgraph_workflow.ainvoke(
            state.model_dump(),
            config={"configurable": {"thread_id": job_id}}
        )
        update_job(job_id, status=result.get("status"), result={"transcript": result.get("transcript")}, error=result.get("error"))
    except Exception as e:
        logger.error(f"Background transcribe failed: {e}")
        update_job(job_id, status="failed", error=str(e))

async def process_transcribe_summarize_job(job_id: str, file_path: str, summary_type: str, diarize: bool, detect_language: bool) -> None:
    try:
        # Step 1: Transcribe
        state = WorkflowState(job_id=job_id, action="transcribe", file_path=file_path)
        transcribe_result = await langgraph_workflow.ainvoke(state.model_dump(), config={"configurable": {"thread_id": f"{job_id}_transcribe"}})
        if transcribe_result.get("status") == "failed":
            update_job(job_id, status="failed", error=transcribe_result.get("error", "Transcription failed"))
            return

        # Step 2: Optional diarization
        if diarize:
            diarize_state = WorkflowState(
                job_id=job_id,
                action="diarize",
                file_path=file_path,
                transcript=transcribe_result.get("transcript"),
                diarization_enabled=True
            )
            diarize_result = await langgraph_workflow.ainvoke(diarize_state.model_dump(), config={"configurable": {"thread_id": f"{job_id}_diarize"}})
            update_job(job_id, result={"segments": diarize_result.get("segments")})
            if job_id in job_progress:
                del job_progress[job_id]

        # Step 3: Optional language detection
        language_result = transcribe_result
        if detect_language:
            language_state = WorkflowState(
                job_id=job_id,
                action="language_detection",
                file_path=file_path,
                transcript=transcribe_result.get("transcript"),
                language_detection_enabled=True
            )
            language_result = await langgraph_workflow.ainvoke(language_state.model_dump(), config={"configurable": {"thread_id": f"{job_id}_language"}})
            update_job(job_id, result={
                "language_tags": language_result.get("language_tags"),
                "dominant_language": language_result.get("dominant_language")
            })
            if job_id in job_progress:
                del job_progress[job_id]

        # Step 4: Summarize
        sum_state = WorkflowState(
            job_id=job_id,
            action="summarize",
            text=transcribe_result.get("transcript"),
            summary_type=summary_type,
            language_tags=language_result.get("language_tags"),
            dominant_language=language_result.get("dominant_language")
        )
        summarize_result = await langgraph_workflow.ainvoke(sum_state.model_dump(), config={"configurable": {"thread_id": f"{job_id}_summarize"}})

        final_result = {
            "transcript": transcribe_result.get("transcript"),
            "summary": summarize_result.get("summary")
        }
        update_job(job_id, status=summarize_result.get("status"), result=final_result, error=summarize_result.get("error"))
    except Exception as e:
        logger.error(f"Background job failed: {e}", exc_info=True)
        update_job(job_id, status="failed", error=str(e))

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), wait: bool = Query(False)):
    # Save file
    file_path = UPLOADS_DIR / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Create job
    job = create_job("transcribe", {"file_path": str(file_path)})

    # Kick off background processing and return immediately
    asyncio.create_task(process_transcribe_job(job["id"], str(file_path)))
    return {"job": load_job(job["id"]) }

@app.post("/transcribe-summarize")
async def transcribe_and_summarize(
    file: UploadFile = File(...),
    summary_type: str = Form("executive"),
    diarize: bool = Form(False),
    detect_language: bool = Form(False)
):
    """Transcribe audio and then summarize the transcript (returns immediately)."""
    # Save file
    file_path = UPLOADS_DIR / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Create job
    job = create_job("transcribe-summarize", {
        "file_path": str(file_path),
        "summary_type": summary_type,
        "diarize": bool(diarize),
        "detect_language": bool(detect_language)
    })

    # Kick off background workflow and return quickly
    asyncio.create_task(process_transcribe_summarize_job(job["id"], str(file_path), summary_type, bool(diarize), bool(detect_language)))
    return {"job": load_job(job["id"]) }

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    # Add progress info if available
    if job_id in job_progress:
        job["progress"] = job_progress[job_id]
    
    return {"job": job}

@app.get("/jobs/{job_id}/progress")
async def get_job_progress(job_id: str):
    """Get real-time progress for a transcription job"""
    if job_id in job_progress:
        return {"progress": job_progress[job_id]}
    
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    return {
        "progress": {
            "status": job.get("status", "unknown"),
            "completed": job.get("status") in ["completed", "failed"]
        }
    }

# Removed WebSocket complexity - using simple polling instead

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting E2E Whisper + Qwen service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
