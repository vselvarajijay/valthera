#!/usr/bin/env python3

"""
Streamlit dashboard for vehicle tracking.

This is the main Streamlit application that provides a web interface
for viewing vehicle detections, timelines, and managing tracking.
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import os
import base64
import io
from PIL import Image
import time
import websocket
import threading

# Configure page
st.set_page_config(
    page_title="🚗👥 Object Tracker",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://jarvis-api:8001/api/v1"

def main():
    """Main Streamlit application"""
    
    # Sidebar navigation
    st.sidebar.title("🚗👥 Object Tracker")
    
    # Object type selection
    st.sidebar.markdown("### 🎯 Object Type")
    object_type = st.sidebar.radio(
        "Track",
        ["🚗 Vehicles", "👥 People"],
        help="Select object type to track (only one active at a time)"
    )
    
    # Extract object type and API endpoints
    if object_type == "🚗 Vehicles":
        object_name = "vehicles"
        object_icon = "🚗"
        api_prefix = "/vehicles"
    else:
        object_name = "people"
        object_icon = "👥"
        api_prefix = "/people"
    
    # Configuration debug
    st.sidebar.markdown("### ⚙️ Configuration")
    st.sidebar.code(f"API_BASE_URL: {API_BASE_URL}")
    st.sidebar.code(f"Object Type: {object_name}")
    st.sidebar.code(f"Environment: Docker")
    
    page = st.sidebar.selectbox(
        "Navigate",
        ["🏠 Dashboard", f"{object_icon} Gallery", "📊 Timeline", "🎥 Playback", "📹 Live View", "⚙️ Settings"]
    )
    
    # API status check
    try:
        response = requests.get(f"{API_BASE_URL}{api_prefix}/stats", timeout=5)
        api_status = "🟢 Connected" if response.status_code == 200 else "🔴 Error"
        
        # Debug information
        st.sidebar.markdown("### 🔍 Debug Info")
        st.sidebar.code(f"API URL: {API_BASE_URL}{api_prefix}/stats")
        st.sidebar.code(f"Status Code: {response.status_code}")
        st.sidebar.code(f"Response: {response.text[:100]}...")
        
    except Exception as e:
        api_status = "🔴 Disconnected"
        
        # Debug information for connection errors
        st.sidebar.markdown("### 🔍 Debug Info")
        st.sidebar.code(f"API URL: {API_BASE_URL}{api_prefix}/stats")
        st.sidebar.code(f"Error: {str(e)}")
        st.sidebar.code(f"Error Type: {type(e).__name__}")
    
    st.sidebar.markdown(f"**API Status:** {api_status}")
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_dashboard(object_name, object_icon, api_prefix)
    elif page == f"{object_icon} Gallery":
        show_gallery(object_name, object_icon, api_prefix)
    elif page == "📊 Timeline":
        show_timeline(object_name, object_icon, api_prefix)
    elif page == "🎥 Playback":
        show_playback(object_name, object_icon, api_prefix)
    elif page == "📹 Live View":
        show_live_view()
    elif page == "⚙️ Settings":
        show_settings(object_name, object_icon, api_prefix)


def show_dashboard(object_name, object_icon, api_prefix):
    """Show main dashboard with statistics"""
    st.title(f"{object_icon} {object_name.title()} Tracking Dashboard")
    
    # Debug section
    st.subheader("🔍 Debug Information")
    debug_col1, debug_col2 = st.columns(2)
    
    with debug_col1:
        st.markdown("**API Request Details:**")
        st.code(f"URL: {API_BASE_URL}{api_prefix}/stats")
        st.code(f"Method: GET")
        st.code(f"Timeout: 5 seconds")
    
    try:
        # Get object statistics
        st.markdown("**Making API Request...**")
        response = requests.get(f"{API_BASE_URL}{api_prefix}/stats", timeout=5)
        
        with debug_col2:
            st.markdown("**API Response Details:**")
            st.code(f"Status Code: {response.status_code}")
            st.code(f"Headers: {dict(response.headers)}")
            st.code(f"Response Text: {response.text}")
            st.code(f"Response Length: {len(response.text)} chars")
        
        if response.status_code == 200:
            stats = response.json()
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Map object names to correct field names
            field_mapping = {
                'vehicles': 'vehicles',
                'people': 'persons'
            }
            field_name = field_mapping[object_name]
            
            with col1:
                st.metric(f"Total {object_name.title()}", stats[f'total_{field_name}'])
            
            with col2:
                st.metric(f"Returning {object_name.title()}", stats[f'returning_{field_name}'])
            
            with col3:
                st.metric("Recent (24h)", stats[f'recent_{field_name}'])
            
            with col4:
                st.metric("Return Rate", f"{stats['return_rate']:.1f}%")
            
            # Additional stats
            st.subheader("📈 Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                # Map to singular for avg_sightings field
                singular_mapping = {
                    'vehicles': 'vehicle',
                    'people': 'person'
                }
                singular_name = singular_mapping[object_name]
                st.metric(f"Avg Sightings/{object_name.title()}", f"{stats[f'avg_sightings_per_{singular_name}']:.1f}")
            
            with col2:
                # Get tracking status
                try:
                    status_response = requests.get(f"{API_BASE_URL}/tracking/status")
                    if status_response.status_code == 200:
                        tracking_status = status_response.json()
                        current_mode = tracking_status.get('current_mode', 'stopped')
                        is_running = tracking_status.get('is_running', False)
                        
                        if is_running:
                            status_text = f"🟢 Running ({current_mode.title()})"
                        else:
                            status_text = "🔴 Stopped"
                        st.metric("Tracking Status", status_text)
                except:
                    st.metric("Tracking Status", "❓ Unknown")
            
            # Recent activity
            st.subheader("🕒 Recent Activity")
            try:
                objects_response = requests.get(f"{API_BASE_URL}{api_prefix}/")
                if objects_response.status_code == 200:
                    objects = objects_response.json()
                    
                    # Show recent objects
                    recent_objects = sorted(objects, key=lambda x: x['last_seen'], reverse=True)[:5]
                    
                    for obj in recent_objects:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            st.write(f"**{object_name.title()} #{obj['object_id']}**")
                        with col2:
                            st.write(f"Last seen: {obj['last_seen']}")
                        with col3:
                            st.write(f"Seen {obj['count']} times")
                        
                        st.divider()
            
            except Exception as e:
                st.error(f"Error loading recent activity: {e}")
        
        else:
            st.error(f"Failed to load {object_name} statistics")
            st.markdown("**Error Details:**")
            st.code(f"Status Code: {response.status_code}")
            st.code(f"Response: {response.text}")
            st.code(f"URL: {response.url}")
    
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        st.markdown("**Exception Details:**")
        st.code(f"Exception Type: {type(e).__name__}")
        st.code(f"Exception Message: {str(e)}")
        st.code(f"Request URL: {API_BASE_URL}{api_prefix}/stats")
        
        # Additional debugging for common issues
        if "ConnectionError" in str(type(e)):
            st.warning("🔗 **Connection Error**: The API server might be down or unreachable")
        elif "Timeout" in str(type(e)):
            st.warning("⏱️ **Timeout Error**: The API server is taking too long to respond")
        elif "HTTPError" in str(type(e)):
            st.warning("🌐 **HTTP Error**: The API server returned an error status")


def show_gallery(object_name, object_icon, api_prefix):
    """Show object gallery with thumbnails"""
    st.title(f"{object_icon} {object_name.title()} Gallery")
    
    try:
        # Get all objects
        response = requests.get(f"{API_BASE_URL}{api_prefix}/")
        if response.status_code == 200:
            objects = response.json()
            
            if not objects:
                st.info(f"No {object_name} detected yet. Start tracking to see {object_name} here!")
                return
            
            # Display objects in grid
            cols = st.columns(4)
            
            for i, obj in enumerate(objects):
                with cols[i % 4]:
                    # Display thumbnail if available
                    if obj['thumbnail_paths']:
                        try:
                            thumbnail_path = obj['thumbnail_paths'][0]
                            if os.path.exists(thumbnail_path):
                                st.image(thumbnail_path, width=200)
                            else:
                                st.image("https://via.placeholder.com/200x150/cccccc/666666?text=No+Image", width=200)
                        except:
                            st.image("https://via.placeholder.com/200x150/cccccc/666666?text=No+Image", width=200)
                    else:
                        st.image("https://via.placeholder.com/200x150/cccccc/666666?text=No+Image", width=200)
                    
                    st.markdown(f"**{object_name.title()} #{obj['object_id']}**")
                    st.caption(f"Seen {obj['count']} times")
                    st.caption(f"Last seen: {obj['last_seen']}")
                    
                    # Button to view timeline
                    if st.button(f"View Timeline", key=f"timeline_{obj['object_id']}"):
                        st.session_state[f'selected_{object_name}'] = obj['object_id']
                        st.rerun()
            
            # Show selected object info
            if f'selected_{object_name}' in st.session_state:
                st.subheader(f"Selected {object_name.title()} #{st.session_state[f'selected_{object_name}']}")
                
                # Get detailed object info
                try:
                    obj_response = requests.get(f"{API_BASE_URL}{api_prefix}/{st.session_state[f'selected_{object_name}']}")
                    if obj_response.status_code == 200:
                        obj = obj_response.json()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**First Seen:** {obj['first_seen']}")
                            st.write(f"**Last Seen:** {obj['last_seen']}")
                            st.write(f"**Total Sightings:** {obj['count']}")
                        
                        with col2:
                            st.write(f"**Thumbnails:** {len(obj['thumbnail_paths'])}")
                            st.write(f"**Video Clips:** {len(obj['clip_paths'])}")
                        
                        # Show all thumbnails
                        if obj['thumbnail_paths']:
                            st.subheader("📸 All Sightings")
                            thumbnail_cols = st.columns(4)
                            for i, thumb_path in enumerate(obj['thumbnail_paths']):
                                with thumbnail_cols[i % 4]:
                                    if os.path.exists(thumb_path):
                                        st.image(thumb_path, width=150)
                                    else:
                                        st.image("https://via.placeholder.com/150x100/cccccc/666666?text=Missing", width=150)
                
                except Exception as e:
                    st.error(f"Error loading {object_name} details: {e}")
        
        else:
            st.error(f"Failed to load {object_name}")
    
    except Exception as e:
        st.error(f"Error connecting to API: {e}")


def show_timeline(object_name, object_icon, api_prefix):
    """Show timeline for selected object"""
    st.title(f"📊 {object_name.title()} Timeline")
    
    if f'selected_{object_name}' not in st.session_state:
        st.info(f"Please select a {object_name[:-1]} from the Gallery to view its timeline.")
        return
    
    object_id = st.session_state[f'selected_{object_name}']
    
    try:
        # Get object timeline
        response = requests.get(f"{API_BASE_URL}{api_prefix}/{object_id}")
        if response.status_code == 200:
            obj = response.json()
            
            st.subheader(f"{object_name.title()} #{object_id} Timeline")
            
            # Object summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sightings", obj['count'])
            with col2:
                st.metric("First Seen", obj['first_seen'][:10])
            with col3:
                st.metric("Last Seen", obj['last_seen'][:10])
            
            # Timeline visualization
            st.subheader("📅 Detection Timeline")
            
            # Show sightings chronologically
            sightings = []
            for i, (thumb_path, clip_path) in enumerate(zip(obj['thumbnail_paths'], obj['clip_paths'])):
                sightings.append({
                    'index': i + 1,
                    'thumbnail': thumb_path,
                    'clip': clip_path,
                    'timestamp': f"Sighting #{i + 1}"
                })
            
            # Display timeline
            for sighting in sightings:
                st.markdown(f"### {sighting['timestamp']}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if os.path.exists(sighting['thumbnail']):
                        st.image(sighting['thumbnail'], width=200)
                    else:
                        st.image("https://via.placeholder.com/200x150/cccccc/666666?text=Missing", width=200)
                
                with col2:
                    st.write(f"**Detection #{sighting['index']}**")
                    if os.path.exists(sighting['clip']):
                        st.video(sighting['clip'])
                    else:
                        st.info("Video clip not available")
                
                st.divider()
        
        else:
            st.error(f"Failed to load {object_name} timeline")
    
    except Exception as e:
        st.error(f"Error loading timeline: {e}")


def show_playback(object_name, object_icon, api_prefix):
    """Show video playback interface"""
    st.title("🎥 Video Playback")
    
    if f'selected_{object_name}' not in st.session_state:
        st.info(f"Please select a {object_name[:-1]} from the Gallery to view its videos.")
        return
    
    object_id = st.session_state[f'selected_{object_name}']
    
    try:
        # Get object clips
        response = requests.get(f"{API_BASE_URL}{api_prefix}/{object_id}/clips")
        if response.status_code == 200:
            clips_data = response.json()
            
            st.subheader(f"{object_name.title()} #{object_id} Video Clips")
            
            clips = clips_data['clips']
            if not clips:
                st.info(f"No video clips available for this {object_name[:-1]}.")
                return
            
            # Individual clips
            st.subheader("📹 Individual Clips")
            
            for i, clip_path in enumerate(clips):
                st.markdown(f"**Clip {i + 1}**")
                if os.path.exists(clip_path):
                    st.video(clip_path)
                else:
                    st.error(f"Clip not found: {clip_path}")
                st.divider()
            
            # Merge clips option
            st.subheader("🎬 Merge All Clips")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                output_filename = st.text_input(
                    "Output filename",
                    value=f"{object_name[:-1]}_{object_id}_merged.mp4",
                    help="Name for the merged video file"
                )
            
            with col2:
                if st.button("🔄 Generate Merged Video", type="primary"):
                    with st.spinner("Merging clips..."):
                        try:
                            merge_response = requests.post(
                                f"{API_BASE_URL}{api_prefix}/{object_id}/merge",
                                json={"output_filename": output_filename}
                            )
                            
                            if merge_response.status_code == 200:
                                merge_result = merge_response.json()
                                st.success("Video merged successfully!")
                                st.json(merge_result)
                                
                                # Show merged video
                                merged_path = merge_result['merged_video_path']
                                if os.path.exists(merged_path):
                                    st.subheader("🎬 Merged Video")
                                    st.video(merged_path)
                            
                            else:
                                st.error(f"Failed to merge clips: {merge_response.text}")
                        
                        except Exception as e:
                            st.error(f"Error merging clips: {e}")
        
        else:
            st.error(f"Failed to load {object_name} clips")
    
    except Exception as e:
        st.error(f"Error loading playback: {e}")


def show_settings(object_name, object_icon, api_prefix):
    """Show tracking settings and configuration"""
    st.title("⚙️ Settings")
    
    try:
        # Get current tracking status and config
        status_response = requests.get(f"{API_BASE_URL}{api_prefix}/tracking/status")
        config_response = requests.get(f"{API_BASE_URL}{api_prefix}/tracking/config")
        
        if status_response.status_code == 200 and config_response.status_code == 200:
            status = status_response.json()
            config = config_response.json()
            
            # Unified tracking control
            st.subheader("🎮 Tracking Control")
            
            # Get unified tracking status
            try:
                unified_status_response = requests.get(f"{API_BASE_URL}/tracking/status")
                if unified_status_response.status_code == 200:
                    unified_status = unified_status_response.json()
                    current_mode = unified_status.get('current_mode', 'stopped')
                    is_running = unified_status.get('is_running', False)
                    
                    # Display current status with debugging
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        if is_running:
                            status_text = f"🟢 Running ({current_mode.title()})"
                        else:
                            status_text = "🔴 Stopped"
                        st.metric("Current Status", status_text)
                    
                    with col2:
                        if st.button("⏹️ Stop", type="secondary"):
                            try:
                                stop_response = requests.post(f"{API_BASE_URL}/tracking/stop")
                                if stop_response.status_code == 200:
                                    st.success("Tracking stopped")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to stop tracking: {stop_response.status_code}")
                                    st.code(f"Response: {stop_response.text}")
                            except Exception as e:
                                st.error(f"Error stopping tracking: {e}")
                    
                    with col3:
                        if st.button("▶️ Start", type="primary"):
                            try:
                                # Start tracking for the current object type
                                start_response = requests.post(
                                    f"{API_BASE_URL}/tracking/start",
                                    json={"object_type": object_name}
                                )
                                if start_response.status_code == 200:
                                    st.success(f"Tracking started for {object_name}")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to start tracking: {start_response.status_code}")
                                    st.code(f"Response: {start_response.text}")
                            except Exception as e:
                                st.error(f"Error starting tracking: {e}")
                    
                    # Mode switching controls
                    st.subheader("🔄 Switch Tracking Mode")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🚗 Switch to Vehicles", disabled=(current_mode == 'vehicles')):
                            try:
                                switch_response = requests.post(
                                    f"{API_BASE_URL}/tracking/switch",
                                    json={"object_type": "vehicles"}
                                )
                                if switch_response.status_code == 200:
                                    st.success("Switched to vehicle tracking")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to switch: {switch_response.status_code}")
                                    st.code(f"Response: {switch_response.text}")
                            except Exception as e:
                                st.error(f"Error switching to vehicles: {e}")
                    
                    with col2:
                        if st.button("👥 Switch to People", disabled=(current_mode == 'people')):
                            try:
                                switch_response = requests.post(
                                    f"{API_BASE_URL}/tracking/switch",
                                    json={"object_type": "people"}
                                )
                                if switch_response.status_code == 200:
                                    st.success("Switched to people tracking")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to switch: {switch_response.status_code}")
                                    st.code(f"Response: {switch_response.text}")
                            except Exception as e:
                                st.error(f"Error switching to people: {e}")
                    
                    # Debug information
                    with st.expander("🔍 Debug Information"):
                        st.json(unified_status)
                        
                        # Test API endpoints
                        st.subheader("API Endpoint Tests")
                        
                        test_endpoints = [
                            ("Unified Status", f"{API_BASE_URL}/tracking/status"),
                            ("People Stats", f"{API_BASE_URL}/people/stats"),
                            ("Vehicles Stats", f"{API_BASE_URL}/vehicles/stats"),
                        ]
                        
                        for name, url in test_endpoints:
                            try:
                                response = requests.get(url, timeout=5)
                                st.write(f"**{name}**: {response.status_code} ✅")
                                if response.status_code != 200:
                                    st.code(f"Error: {response.text}")
                            except Exception as e:
                                st.write(f"**{name}**: ❌ {e}")
                
                else:
                    st.error(f"Failed to get unified tracking status: {unified_status_response.status_code}")
                    st.code(f"Response: {unified_status_response.text}")
                    
            except Exception as e:
                st.error(f"Error getting unified tracking status: {e}")
            
            # Configuration
            st.subheader("⚙️ Tracking Configuration")
            
            with st.form("config_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    sample_interval = st.number_input(
                        "Sample Interval (seconds)",
                        min_value=1.0,
                        max_value=60.0,
                        value=float(config['sample_interval_seconds']),
                        step=0.5,
                        help="How often to process frames"
                    )
                    
                    similarity_threshold = st.slider(
                        "Similarity Threshold",
                        min_value=0.1,
                        max_value=1.0,
                        value=float(config['similarity_threshold']),
                        step=0.05,
                        help="Threshold for matching objects"
                    )
                
                with col2:
                    max_detections = st.number_input(
                        "Max Detections per Frame",
                        min_value=1,
                        max_value=50,
                        value=int(config['max_detections_per_frame']),
                        help="Maximum detections to process per frame"
                    )
                    
                    cleanup_days = st.number_input(
                        "Cleanup Days",
                        min_value=1,
                        max_value=30,
                        value=int(config['cleanup_days_old']),
                        help="Days to keep old clips"
                    )
                
                if st.form_submit_button("💾 Update Configuration", type="primary"):
                    try:
                        update_data = {
                            "sample_interval_seconds": sample_interval,
                            "similarity_threshold": similarity_threshold,
                            "max_detections_per_frame": max_detections,
                            "cleanup_days_old": cleanup_days
                        }
                        
                        update_response = requests.post(
                            f"{API_BASE_URL}{api_prefix}/tracking/config",
                            json=update_data
                        )
                        
                        if update_response.status_code == 200:
                            st.success("Configuration updated successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to update configuration")
                    
                    except Exception as e:
                        st.error(f"Error updating configuration: {e}")
            
            # Storage management
            st.subheader("💾 Storage Management")
            
            try:
                storage_response = requests.get(f"{API_BASE_URL}{api_prefix}/storage/stats")
                
                if storage_response.status_code == 200:
                    storage_stats = storage_response.json()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Clips", storage_stats['total_clips'])
                    with col2:
                        st.metric("Merged Videos", storage_stats['total_merged'])
                    with col3:
                        st.metric("Total Size", f"{storage_stats['total_size_mb']:.1f} MB")
                    
                    # Cleanup options
                    st.subheader("🧹 Storage Cleanup")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        cleanup_days = st.number_input(
                            "Delete files older than (days)",
                            min_value=1,
                            max_value=30,
                            value=7,
                            key="cleanup_days"
                        )
                    
                    with col2:
                        dry_run = st.checkbox("Dry run (preview only)", value=True)
                    
                    if st.button("🧹 Run Cleanup"):
                        with st.spinner("Running cleanup..."):
                            try:
                                cleanup_response = requests.post(
                                    f"{API_BASE_URL}{api_prefix}/storage/cleanup",
                                    params={
                                        "days_old": cleanup_days,
                                        "dry_run": dry_run
                                    }
                                )
                                
                                if cleanup_response.status_code == 200:
                                    cleanup_result = cleanup_response.json()
                                    st.success(cleanup_result['message'])
                                    st.json(cleanup_result['stats'])
                                else:
                                    st.error("Failed to run cleanup")
                            
                            except Exception as e:
                                st.error(f"Error running cleanup: {e}")
                
                else:
                    st.error("Failed to load storage statistics")
            
            except Exception as e:
                st.error(f"Error loading storage info: {e}")
        
        else:
            st.error("Failed to load tracking configuration")
    
    except Exception as e:
        st.error(f"Error loading settings: {e}")


def show_live_view():
    """Show live camera feed with tracking overlays"""
    st.title("📹 Live Camera View")
    
    # Initialize session state
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'last_frame_time' not in st.session_state:
        st.session_state.last_frame_time = 0
    
    # Get current tracking status
    tracking_status = None
    try:
        tracking_response = requests.get(f"{API_BASE_URL}/tracking/status", timeout=2)
        if tracking_response.status_code == 200:
            tracking_status = tracking_response.json()
    except:
        pass
    
    # Connection status and tracking info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Try to fetch annotated frame first (with bounding boxes)
        annotated_available = False
        try:
            response = requests.get(f"{API_BASE_URL}/live/annotated", timeout=2)
            if response.status_code == 200:
                st.success("🟢 Camera Connected (Annotated)")
                annotated_available = True
                camera_connected = True
            else:
                # Fall back to raw frame
                response = requests.get(f"{API_BASE_URL}/live/raw", timeout=2)
                if response.status_code == 200:
                    st.success("🟢 Camera Connected (Raw)")
                    camera_connected = True
                else:
                    st.error("🔴 Camera Disconnected")
                    camera_connected = False
        except:
            st.error("🔴 Camera Disconnected")
            camera_connected = False
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    with col3:
        if tracking_status:
            current_mode = tracking_status.get('current_mode', 'stopped')
            is_running = tracking_status.get('is_running', False)
            if is_running:
                st.info(f"🎯 Tracking: {current_mode.title()}")
            else:
                st.warning("⏹️ Tracking: Stopped")
    
    # Display live feed
    st.subheader("📷 Live Camera Feed")
    
    if camera_connected:
        try:
            # Try annotated frame first, fall back to raw
            if annotated_available:
                response = requests.get(f"{API_BASE_URL}/live/annotated", timeout=2)
                frame_type = "Annotated (with bounding boxes)"
            else:
                response = requests.get(f"{API_BASE_URL}/live/raw", timeout=2)
                frame_type = "Raw (no bounding boxes)"
            
            if response.status_code == 200:
                # Display frame
                image = Image.open(io.BytesIO(response.content))
                st.image(image, width='stretch', caption=f"📸 {frame_type}")
                
                # Update metrics
                st.session_state.frame_count += 1
                current_time = time.time()
                
                # Frame information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Frame Count", st.session_state.frame_count)
                
                with col2:
                    if st.session_state.last_frame_time > 0:
                        fps = 1.0 / max(0.01, current_time - st.session_state.last_frame_time)
                        st.metric("FPS", f"{fps:.1f}")
                    else:
                        st.metric("FPS", "0.0")
                
                with col3:
                    st.metric("Status", "🟢 Live")
                
                st.session_state.last_frame_time = current_time
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.subheader("Tracking Status")
                    if tracking_status:
                        st.json(tracking_status)
                    else:
                        st.warning("No tracking status available")
                    
                    st.subheader("API Endpoint Tests")
                    test_endpoints = [
                        ("Annotated Frame", f"{API_BASE_URL}/live/annotated"),
                        ("Raw Frame", f"{API_BASE_URL}/live/raw"),
                        ("Live View Status", f"{API_BASE_URL}/live/status"),
                        ("Tracking Status", f"{API_BASE_URL}/tracking/status"),
                        ("Pipeline Status", f"{API_BASE_URL}/pipeline/status"),
                    ]
                    
                    for name, url in test_endpoints:
                        try:
                            response = requests.get(url, timeout=2)
                            st.write(f"**{name}**: {response.status_code} ✅")
                            if response.status_code != 200:
                                st.code(f"Error: {response.text}")
                        except Exception as e:
                            st.write(f"**{name}**: ❌ {e}")
                
                # Auto-refresh for smooth video
                time.sleep(0.1)
                st.rerun()
            else:
                st.error("Failed to fetch camera frame")
        except Exception as e:
            st.error(f"Error fetching camera feed: {e}")
    else:
        st.info("Waiting for camera feed...")
        placeholder_image = Image.new('RGB', (640, 480), color='lightgray')
        st.image(placeholder_image, width='stretch')
        
        # Auto-refresh to check for connection
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()
