#!/usr/bin/env python3
"""
Streamlit UI for Audio Transcription & Summarization
"""
import streamlit as st
import httpx
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict
import json
import os
# Removed websockets - using simple polling instead

# Configure page
st.set_page_config(
    page_title="Audio Transcription & Summarization",
    page_icon="🎙️",
    layout="wide"
)

# API endpoint (configurable via env)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Language code to name mapping
LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'Hindi', 
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'mr': 'Marathi',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese',
    'ur': 'Urdu',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'my': 'Myanmar',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'tl': 'Filipino',
    'sw': 'Swahili',
    'am': 'Amharic',
    'he': 'Hebrew',
    'tr': 'Turkish',
    'pl': 'Polish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'cs': 'Czech',
    'sk': 'Slovak',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
    'hr': 'Croatian',
    'sr': 'Serbian',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'mt': 'Maltese',
    'ga': 'Irish',
    'cy': 'Welsh',
    'eu': 'Basque',
    'ca': 'Catalan',
    'gl': 'Galician',
    'unknown': 'Unknown'
}

def _display_language_detection_box(language_info):
    """Display a prominent language detection results box"""
    dominant_language = language_info.get('dominant_language', 'unknown')
    detected_languages = language_info.get('detected_languages', [])
    
    # Get language names
    dominant_name = LANGUAGE_NAMES.get(dominant_language, dominant_language.upper())
    detected_names = [LANGUAGE_NAMES.get(lang, lang.upper()) for lang in detected_languages]
    
    # Create the language detection box
    st.markdown("### 🌍 Language Detection Results")
    
    # Main language display
    if len(detected_languages) == 1:
        # Single language detected
        lang_emoji = "🇺🇸" if dominant_language == "en" else "🇮🇳" if dominant_language == "hi" else "🗣️"
        st.success(f"**{lang_emoji} Detected Language: {dominant_name}**")
        st.info("ℹ️ This audio contains primarily **single-language** content.")
    
    elif len(detected_languages) > 1:
        # Multiple languages detected - this is a mix!
        st.warning(f"**🌐 Mixed Language Content Detected!**")
        
        # Show dominant language
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Primary Language", dominant_name)
        with col2:
            st.metric("Total Languages", f"{len(detected_languages)} languages")
        
        # Show all detected languages
        languages_str = " • ".join(detected_names)
        st.info(f"**Languages found:** {languages_str}")
        
        # Special message for common Indian language mixes
        indian_langs = {'hi', 'ta', 'te', 'kn', 'ml', 'bn', 'gu', 'mr', 'pa', 'or', 'as', 'ur'}
        detected_set = set(detected_languages)
        
        if 'en' in detected_set and detected_set.intersection(indian_langs):
            st.success("🇮🇳 **Code-switching detected!** This appears to be Indian multilingual content with English mixed in.")
        elif len(detected_set.intersection(indian_langs)) > 1:
            st.success("🇮🇳 **Multi-Indian language content** detected!")
        else:
            st.success("🌏 **International multilingual content** detected!")
    
    else:
        # No languages detected or unknown
        st.error("❓ **Language could not be determined**")
        st.caption("The audio might be too short, unclear, or in an unsupported language.")
    
    # Add a small note about language-aware processing
    if len(detected_languages) > 0:
        st.caption("💡 The summary above was generated using language-aware prompts optimized for the detected language(s).")

# Initialize session state
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def submit_job(file_path: str, summary_type: str = "executive", diarize: bool = False, detect_language: bool = False):
    """Submit audio file for transcription and summarization (synchronous, fast return)."""
    with httpx.Client(timeout=30.0) as client:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'audio/mpeg')}
            data = {
                "summary_type": summary_type,
                "diarize": str(bool(diarize)).lower(),
                "detect_language": str(bool(detect_language)).lower()
            }
            response = client.post(
                f"{API_URL}/transcribe-summarize",
                files=files,
                data=data
            )
        response.raise_for_status()
        return response.json()['job']

async def get_job_progress(job_id: str):
    """Get job progress"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/jobs/{job_id}/progress")
        if response.status_code == 200:
            return response.json()['progress']
    return None

async def get_job_result(job_id: str):
    """Get job result"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/jobs/{job_id}")
        if response.status_code == 200:
            return response.json()['job']
    return None

def main():
    st.title("🎙️ Audio Transcription & Summarization")
    st.markdown("Upload an audio file to get an AI-powered transcript and summary")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        summary_type = st.selectbox(
            "Summary Type",
            ["executive", "detailed", "bullet_points"],
            help="Choose the type of summary you want"
        )
        diarize = st.toggle(
            "Enable Speaker Detection (Diarization)", value=False,
            help="Identify speakers in the transcript"
        )
        
        detect_language = st.toggle(
            "Enable Language Detection", value=False,
            help="Detect languages in the transcript for better summarization"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses:
        - **OpenAI Whisper** for transcription
        - **Qwen 2.5** for summarization
        - **LangGraph** for workflow orchestration
        """)
        
        st.markdown("---")
        st.markdown("### Performance Tips")
        st.markdown("""
        - Disable diarization for 2x faster processing
        - Files under 1 min process in ~15-30s
        - Files 5+ min may take 1-2 minutes
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Audio")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'm4a', 'ogg', 'webm'],
            help="Maximum file size: 200MB"
        )
        
        if uploaded_file and not st.session_state.processing:
            st.success(f"✓ {uploaded_file.name} ready to process")
            
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.session_state.current_job_id = None
                st.rerun()
    
    with col2:
        if st.session_state.processing and uploaded_file:
            st.header("⚙️ Processing")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            # Submit job
            try:
                status_line = st.empty()
                status_line.info("📤 Uploading audio file and creating job...")
                job = submit_job(temp_path, summary_type, diarize, detect_language)
                st.session_state.current_job_id = job['id']
                status_line.success("✅ Job created. Processing started.")
                
                # Show processing stages with detailed progress
                if st.session_state.current_job_id:
                    st.markdown("### Processing Stages")
                    
                    # Create placeholders for each stage
                    stage_container = st.container()
                    with stage_container:
                        transcribe_status = st.empty()
                        diarize_status = st.empty() if diarize else None
                        language_status = st.empty() if detect_language else None
                        summary_status = st.empty()
                        
                        # Initial status display
                        transcribe_status.info("🎙️ **Step 1/3:** Transcribing audio...")
                        if diarize:
                            diarize_status.info("⏸️ **Step 2/4:** Waiting to diarize...")
                        if detect_language:
                            language_status.info("⏸️ **Step 2/4:** Waiting for language detection...")
                        summary_status.info("⏸️ **Step 3/3:** Waiting to summarize...")
                    
                    # Progress bar
                    overall_progress = st.progress(0)
                    progress_text = st.empty()
                    
                    # Poll for progress
                    display_detailed_progress(
                        st.session_state.current_job_id,
                        overall_progress,
                        progress_text,
                        transcribe_status,
                        diarize_status,
                        language_status,
                        summary_status
                    )
                
                # Get final result
                result = asyncio.run(get_job_result(st.session_state.current_job_id))
                
                # Final status line
                st.info(f"Status: {result.get('status', 'unknown')}")

                if result['status'] == 'completed':
                    # Success message already shown by progress tracking
                    pass
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["📝 Summary", "📄 Full Transcript", "📊 Job Details"])
                    
                    with tab1:
                        st.markdown("### Summary")
                        if result['result']['summary']:
                            st.markdown(result['result']['summary']['content'])
                        else:
                            st.warning("No summary generated")
                    
                    with tab2:
                        st.markdown("### Full Transcript")
                        if result['result']['transcript']:
                            st.text_area(
                                "Transcript",
                                result['result']['transcript'],
                                height=400,
                                disabled=True
                            )
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Transcript",
                                data=result['result']['transcript'],
                                file_name=f"transcript_{st.session_state.current_job_id}.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("No transcript generated")
                        
                        # Language Detection Box (if available)
                        summary_data = result['result'].get('summary', {})
                        language_info = summary_data.get('language_info') if isinstance(summary_data, dict) else None
                        
                        if language_info and result['inputs'].get('detect_language'):
                            st.markdown("---")
                            _display_language_detection_box(language_info)
                            st.markdown("---")
                        
                        # Speaker segments (if available)
                        segments = None
                        try:
                            segments = result['result'].get('segments')
                        except Exception:
                            segments = None
                        if segments:
                            st.markdown("### Speaker Segments")
                            for seg in segments:
                                start = seg.get('start', 0.0)
                                end = seg.get('end', 0.0)
                                speaker = seg.get('speaker', 'S?')
                                text = seg.get('text') or ''
                                st.write(f"[{start:.1f}–{end:.1f}] {speaker}: {text}")
                    
                    with tab3:
                        st.markdown("### Job Information")
                        
                        # Language information display
                        summary_data = result['result'].get('summary', {})
                        language_info = summary_data.get('language_info') if isinstance(summary_data, dict) else None
                        
                        if language_info:
                            st.markdown("#### Language Detection Results")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                dominant_lang = language_info.get('dominant_language', 'unknown')
                                st.metric("Dominant Language", dominant_lang.upper())
                            
                            with col2:
                                detected_langs = language_info.get('detected_languages', [])
                                if detected_langs:
                                    langs_str = ", ".join([lang.upper() for lang in detected_langs])
                                    st.metric("Detected Languages", f"{len(detected_langs)} languages")
                                    st.caption(langs_str)
                                else:
                                    st.metric("Detected Languages", "None")
                            
                            # Language distribution (if multiple languages detected)
                            if len(detected_langs) > 1:
                                st.info(f"🌍 **Multilingual content detected**: This transcript contains {len(detected_langs)} languages. The summary was generated with language-aware prompts to preserve important terms and context.")
                        
                        st.markdown("#### Technical Details")
                        st.json({
                            "job_id": result['id'],
                            "status": result['status'],
                            "created_at": result['timestamps']['created_at'],
                            "updated_at": result['timestamps']['updated_at'],
                            "transcript_length": len(result['result'].get('transcript', '')),
                            "summary_type": result['inputs'].get('summary_type', 'executive'),
                            "diarization_enabled": result['inputs'].get('diarize', False),
                            "language_detection_enabled": result['inputs'].get('detect_language', False)
                        })
                    
                    if st.button("🔄 Process Another File", use_container_width=True):
                        st.session_state.processing = False
                        st.session_state.current_job_id = None
                        st.rerun()
                
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                    if st.button("🔄 Try Again", use_container_width=True):
                        st.session_state.processing = False
                        st.session_state.current_job_id = None
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if st.button("🔄 Try Again", use_container_width=True):
                    st.session_state.processing = False
                    st.session_state.current_job_id = None
                    st.rerun()
            finally:
                # Clean up temp file
                try:
                    Path(temp_path).unlink()
                except:
                    pass
        
        elif not uploaded_file:
            st.info("👈 Please upload an audio file to begin")

def display_detailed_progress(job_id, overall_progress, progress_text, transcribe_status, diarize_status, language_status, summary_status):
    """Detailed progress display with stage tracking"""
    
    with httpx.Client() as client:
        stages_complete = 0
        total_stages = 3 + (1 if diarize_status else 0) + (1 if language_status else 0)
        
        for _ in range(180):  # Poll for up to 6 minutes (2s intervals)
            try:
                response = client.get(f"{API_URL}/jobs/{job_id}/progress")
                if response.status_code == 200:
                    progress_data = response.json().get("progress", {})
                    status = progress_data.get("status", "unknown")
                    
                    # Update progress based on status
                    if status == "transcribing":
                        completed = progress_data.get("completed_chunks", 0)
                        total = progress_data.get("total_chunks", 1)
                        percentage = progress_data.get("percentage", 0)
                        
                        transcribe_status.warning(f"🎙️ **Step 1:** Transcribing... {completed}/{total} chunks ({percentage}%)")
                        overall_progress.progress(min(0.5, percentage / 200))  # Transcription is 50% of total
                        progress_text.text(f"Processing audio chunks: {completed}/{total}")
                        
                    elif status == "diarizing" and diarize_status:
                        diarize_status.warning("🎯 **Step 2:** Identifying speakers...")
                        transcribe_status.success("✅ **Step 1:** Transcription complete")
                        overall_progress.progress(0.6)
                        progress_text.text("Analyzing speaker patterns...")
                        stages_complete = 1
                        
                    elif status == "language_detection" and language_status:
                        language_status.warning("🌍 **Step 2:** Detecting languages...")
                        transcribe_status.success("✅ **Step 1:** Transcription complete")
                        overall_progress.progress(0.7)
                        progress_text.text("Analyzing language content...")
                        stages_complete = 1
                        
                    elif status == "completed":
                        # Mark all as complete
                        transcribe_status.success("✅ **Step 1:** Transcription complete")
                        if diarize_status:
                            diarize_status.success("✅ **Step 2:** Diarization complete")
                        if language_status:
                            language_status.success("✅ **Step 2:** Language detection complete")
                        summary_status.success("✅ **Step 3:** Summary generated")
                        overall_progress.progress(1.0)
                        progress_text.success("🎉 All processing complete!")
                        break
                        
                    elif status == "failed":
                        error = progress_data.get("error", "Unknown error")
                        overall_progress.progress(0)
                        progress_text.error(f"❌ Processing failed: {error}")
                        break
                        
                    # Check job result for completion
                    job_response = client.get(f"{API_URL}/jobs/{job_id}")
                    if job_response.status_code == 200:
                        job_data = job_response.json().get("job", {})
                        if job_data.get("status") == "completed":
                            # Processing is done
                            transcribe_status.success("✅ **Step 1:** Transcription complete")
                            if diarize_status:
                                diarize_status.success("✅ **Step 2:** Diarization complete")
                            if language_status:
                                language_status.success("✅ **Step 2:** Language detection complete")
                            summary_status.success("✅ **Step 3:** Summary generated")
                            overall_progress.progress(1.0)
                            progress_text.success("🎉 All processing complete!")
                            break
                        
                time.sleep(2)  # Poll every 2 seconds
                
            except Exception as e:
                # Continue polling even if one check fails
                pass

def display_simple_progress(job_id):
    """Fallback simple progress display"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with httpx.Client() as client:
        for _ in range(120):  # Poll for up to 4 minutes
            try:
                response = client.get(f"{API_URL}/jobs/{job_id}")
                if response.status_code == 200:
                    job_data = response.json().get("job", {})
                    status = job_data.get("status", "unknown")
                    
                    if status == "completed":
                        progress_placeholder.success("✅ Processing complete!")
                        status_placeholder.success("✅ All done!")
                        break
                    elif status == "failed":
                        progress_placeholder.error("❌ Processing failed")
                        break
                    else:
                        progress_placeholder.info("🔄 Processing...")
                        
                time.sleep(2)
            except:
                pass

if __name__ == "__main__":
    # Check if API is running
    try:
        response = httpx.get(f"{API_URL}/health", timeout=2.0)
        if response.status_code != 200:
            st.error("❌ API server is not responding. Please ensure the backend is running on port 8000.")
            st.stop()
    except:
        st.error("❌ Cannot connect to API server. Please start the backend first with: `python3 app_e2e.py`")
        st.stop()
    
    main()
