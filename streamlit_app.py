#!/usr/bin/env python3
"""
Streamlit UI for Audio Transcription & Summarization
"""
import streamlit as st
import httpx
import asyncio
import time
from pathlib import Path
import tempfile
import json

# Configure page
st.set_page_config(
    page_title="Audio Transcription & Summarization",
    page_icon="🎙️",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Initialize session state
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

async def transcribe_and_summarize(file_path: str, summary_type: str = "executive"):
    """Submit audio file for transcription and summarization"""
    async with httpx.AsyncClient(timeout=600.0) as client:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'audio/mpeg')}
            data = {'summary_type': summary_type}
            response = await client.post(
                f"{API_URL}/transcribe-summarize",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            return response.json()['job']
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

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
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses:
        - **E2E Networks Whisper** for transcription
        - **Qwen 2.5** for summarization
        - **LangGraph** for workflow orchestration
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
                with st.spinner("Submitting job..."):
                    job = asyncio.run(transcribe_and_summarize(temp_path, summary_type))
                    st.session_state.current_job_id = job['id']
                
                # Progress tracking
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                while True:
                    progress = asyncio.run(get_job_progress(st.session_state.current_job_id))
                    
                    if progress:
                        if 'percentage' in progress:
                            progress_placeholder.progress(
                                progress['percentage'] / 100,
                                text=f"Processing: {progress['completed_chunks']}/{progress['total_chunks']} chunks ({progress['percentage']}%)"
                            )
                        
                        status_placeholder.info(f"Status: {progress.get('status', 'Processing...')}")
                        
                        if progress.get('completed', False):
                            break
                    
                    time.sleep(1)
                
                # Get final result
                result = asyncio.run(get_job_result(st.session_state.current_job_id))
                
                if result['status'] == 'completed':
                    st.success("✓ Processing complete!")
                    
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
                    
                    with tab3:
                        st.markdown("### Job Information")
                        st.json({
                            "job_id": result['id'],
                            "status": result['status'],
                            "created_at": result['timestamps']['created_at'],
                            "updated_at": result['timestamps']['updated_at'],
                            "transcript_length": len(result['result'].get('transcript', '')),
                            "summary_type": result['inputs'].get('summary_type', 'executive')
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
