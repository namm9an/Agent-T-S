#!/usr/bin/env python3
"""
Fast Streamlit UI for Audio Transcription & Summarization
Simplified and optimized for speed
"""
import streamlit as st
import httpx
import time
import tempfile
from pathlib import Path
import os

# Configure page
st.set_page_config(
    page_title="Fast Audio Processing",
    page_icon="🚀",
    layout="wide"
)

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'results' not in st.session_state:
    st.session_state.results = None

def upload_and_process(file_path: str, summary_type: str, diarize: bool) -> dict:
    """Upload file and start processing"""
    with httpx.Client(timeout=30.0) as client:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'audio/mpeg')}
            data = {
                'summary_type': summary_type,
                'diarize': str(diarize).lower()
            }
            
            response = client.post(
                f"{API_URL}/transcribe-summarize",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                return response.json()['job']
            else:
                raise Exception(f"Upload failed: {response.status_code}")

def poll_job(job_id: str, progress_container):
    """Poll job status and update progress"""
    with httpx.Client(timeout=10.0) as client:
        max_polls = 180  # 6 minutes max
        
        for i in range(max_polls):
            try:
                # Get progress
                progress_response = client.get(f"{API_URL}/jobs/{job_id}/progress")
                if progress_response.status_code == 200:
                    progress = progress_response.json()['progress']
                    
                    # Update progress display
                    stage = progress.get('stage', 'processing')
                    message = progress.get('message', 'Processing...')
                    
                    # Show different messages based on stage
                    if stage == 'preparing':
                        progress_container.info(f"🎵 {message}")
                    elif stage == 'transcribing':
                        chunks_done = progress.get('completed_chunks', 0)
                        chunks_total = progress.get('total_chunks', 1)
                        percent = int((chunks_done / max(chunks_total, 1)) * 100)
                        progress_container.progress(percent / 100, f"🎙️ Transcribing: {chunks_done}/{chunks_total} chunks ({percent}%)")
                    elif stage == 'diarizing':
                        progress_container.info(f"👥 {message}")
                    elif stage == 'summarizing':
                        progress_container.info(f"📝 {message}")
                    else:
                        progress_container.info(f"⚙️ {message}")
                    
                    # Check if completed
                    if progress.get('completed', False):
                        break
                
                # Get full job status
                job_response = client.get(f"{API_URL}/jobs/{job_id}")
                if job_response.status_code == 200:
                    job = job_response.json()['job']
                    
                    if job['status'] == 'completed':
                        progress_container.success("✅ Processing complete!")
                        return job
                    elif job['status'] == 'failed':
                        progress_container.error(f"❌ Processing failed: {job.get('error', 'Unknown error')}")
                        return job
                
            except Exception as e:
                # Continue polling even if one request fails
                pass
            
            time.sleep(2)  # Poll every 2 seconds
        
        progress_container.error("⏱️ Processing timeout - please try a shorter audio file")
        return None

def main():
    st.title("🚀 Fast Audio Processing")
    st.markdown("Upload audio → Get transcript & summary in seconds!")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        summary_type = st.selectbox(
            "Summary Type",
            ["executive", "detailed", "bullet_points"],
            help="Choose summary format"
        )
        
        diarize = st.checkbox(
            "Speaker Detection",
            value=False,
            help="Identify different speakers (slower)"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Tips for Speed")
        st.markdown("""
        - Keep diarization OFF for 2x speed
        - Files < 1 min: ~15-20 seconds
        - Files 1-5 min: ~30-60 seconds
        - Files > 5 min: 1-2 minutes
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 Powered by")
        st.markdown("""
        - **OpenAI Whisper** (E2E)
        - **Qwen 2.5** (14B)
        - **FastAPI + AsyncIO**
        """)
    
    # Main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Audio")
        
        uploaded_file = st.file_uploader(
            "Select audio file",
            type=['mp3', 'wav', 'm4a', 'ogg', 'webm', 'mp4'],
            help="Max 200MB"
        )
        
        if uploaded_file:
            st.success(f"✓ {uploaded_file.name} loaded")
            
            # File info
            file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
            uploaded_file.seek(0)  # Reset file pointer
            st.info(f"📁 Size: {file_size_mb:.1f} MB")
            
            if st.button("🚀 Process Audio", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.session_state.results = None
                st.rerun()
    
    with col2:
        if st.session_state.processing and uploaded_file:
            st.header("⚡ Processing")
            
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name
            
            try:
                # Upload and start processing
                with st.spinner("Uploading..."):
                    job = upload_and_process(temp_path, summary_type, diarize)
                    st.session_state.job_id = job['id']
                
                st.success(f"✅ Job created: {job['id'][:8]}...")
                
                # Poll for results with progress display
                progress_container = st.empty()
                result = poll_job(job['id'], progress_container)
                
                if result and result['status'] == 'completed':
                    st.session_state.results = result
                    st.session_state.processing = False
                    st.rerun()
                else:
                    st.error("Processing failed - please try again")
                    st.session_state.processing = False
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.processing = False
            
            finally:
                # Cleanup temp file
                try:
                    Path(temp_path).unlink()
                except:
                    pass
        
        elif st.session_state.results:
            st.header("✨ Results")
            
            result = st.session_state.results
            
            # Display in tabs
            tab1, tab2, tab3 = st.tabs(["📝 Summary", "📄 Transcript", "👥 Speakers"])
            
            with tab1:
                st.markdown("### Summary")
                if result['result']['summary']:
                    st.markdown(result['result']['summary']['content'])
                else:
                    st.warning("No summary generated")
            
            with tab2:
                st.markdown("### Full Transcript")
                transcript = result['result'].get('transcript', '')
                if transcript:
                    # Show in text area
                    st.text_area(
                        "Transcript",
                        transcript,
                        height=400,
                        disabled=True
                    )
                    
                    # Download button
                    st.download_button(
                        "📥 Download Transcript",
                        transcript,
                        f"transcript_{st.session_state.job_id[:8]}.txt",
                        "text/plain"
                    )
                else:
                    st.warning("No transcript generated")
            
            with tab3:
                st.markdown("### Speaker Segments")
                segments = result['result'].get('segments')
                if segments:
                    for seg in segments:
                        speaker = seg.get('speaker', 'Unknown')
                        start = seg.get('start', 0)
                        end = seg.get('end', 0)
                        text = seg.get('text', '')
                        
                        with st.expander(f"{speaker} [{start:.1f}s - {end:.1f}s]"):
                            st.write(text)
                else:
                    st.info("No speaker segments (diarization was disabled)")
            
            # Reset button
            if st.button("🔄 Process Another File", use_container_width=True):
                st.session_state.processing = False
                st.session_state.results = None
                st.session_state.job_id = None
                st.rerun()
        
        else:
            st.info("👈 Upload an audio file to begin")

if __name__ == "__main__":
    # Check API health
    try:
        response = httpx.get(f"{API_URL}/health", timeout=2.0)
        if response.status_code != 200:
            st.error("❌ Backend API is not responding. Please start the backend first.")
            st.code("python3 app_fast.py", language="bash")
            st.stop()
    except:
        st.error("❌ Cannot connect to backend API at http://localhost:8000")
        st.code("python3 app_fast.py", language="bash")
        st.stop()
    
    main()
