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

# Configure page
st.set_page_config(
    page_title="🚗 Vehicle Tracker",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://jarvis-api:8001/api/v1"

def main():
    """Main Streamlit application"""
    
    # Sidebar navigation
    st.sidebar.title("🚗 Vehicle Tracker")
    
    page = st.sidebar.selectbox(
        "Navigate",
        ["🏠 Dashboard", "🚗 Vehicle Gallery", "📊 Timeline", "🎥 Playback", "⚙️ Settings"]
    )
    
    # API status check
    try:
        response = requests.get(f"{API_BASE_URL}/vehicles/stats", timeout=5)
        api_status = "🟢 Connected" if response.status_code == 200 else "🔴 Error"
    except:
        api_status = "🔴 Disconnected"
    
    st.sidebar.markdown(f"**API Status:** {api_status}")
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🚗 Vehicle Gallery":
        show_gallery()
    elif page == "📊 Timeline":
        show_timeline()
    elif page == "🎥 Playback":
        show_playback()
    elif page == "⚙️ Settings":
        show_settings()


def show_dashboard():
    """Show main dashboard with statistics"""
    st.title("🏠 Vehicle Tracking Dashboard")
    
    try:
        # Get vehicle statistics
        response = requests.get(f"{API_BASE_URL}/vehicles/stats")
        if response.status_code == 200:
            stats = response.json()
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Vehicles", stats['total_vehicles'])
            
            with col2:
                st.metric("Returning Vehicles", stats['returning_vehicles'])
            
            with col3:
                st.metric("Recent (24h)", stats['recent_vehicles'])
            
            with col4:
                st.metric("Return Rate", f"{stats['return_rate']:.1f}%")
            
            # Additional stats
            st.subheader("📈 Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Avg Sightings/Vehicle", f"{stats['avg_sightings_per_vehicle']:.1f}")
            
            with col2:
                # Get tracking status
                try:
                    status_response = requests.get(f"{API_BASE_URL}/vehicles/tracking/status")
                    if status_response.status_code == 200:
                        tracking_status = status_response.json()
                        status_text = "🟢 Running" if tracking_status['is_running'] else "🔴 Stopped"
                        st.metric("Tracking Status", status_text)
                except:
                    st.metric("Tracking Status", "❓ Unknown")
            
            # Recent activity
            st.subheader("🕒 Recent Activity")
            try:
                vehicles_response = requests.get(f"{API_BASE_URL}/vehicles/")
                if vehicles_response.status_code == 200:
                    vehicles = vehicles_response.json()
                    
                    # Show recent vehicles
                    recent_vehicles = sorted(vehicles, key=lambda x: x['last_seen'], reverse=True)[:5]
                    
                    for vehicle in recent_vehicles:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            st.write(f"**Vehicle #{vehicle['vehicle_id']}**")
                        with col2:
                            st.write(f"Last seen: {vehicle['last_seen']}")
                        with col3:
                            st.write(f"Seen {vehicle['count']} times")
                        
                        st.divider()
            
            except Exception as e:
                st.error(f"Error loading recent activity: {e}")
        
        else:
            st.error("Failed to load vehicle statistics")
    
    except Exception as e:
        st.error(f"Error connecting to API: {e}")


def show_gallery():
    """Show vehicle gallery with thumbnails"""
    st.title("🚗 Vehicle Gallery")
    
    try:
        # Get all vehicles
        response = requests.get(f"{API_BASE_URL}/vehicles/")
        if response.status_code == 200:
            vehicles = response.json()
            
            if not vehicles:
                st.info("No vehicles detected yet. Start tracking to see vehicles here!")
                return
            
            # Display vehicles in grid
            cols = st.columns(4)
            
            for i, vehicle in enumerate(vehicles):
                with cols[i % 4]:
                    # Display thumbnail if available
                    if vehicle['thumbnail_paths']:
                        try:
                            thumbnail_path = vehicle['thumbnail_paths'][0]
                            if os.path.exists(thumbnail_path):
                                st.image(thumbnail_path, width=200)
                            else:
                                st.image("https://via.placeholder.com/200x150/cccccc/666666?text=No+Image", width=200)
                        except:
                            st.image("https://via.placeholder.com/200x150/cccccc/666666?text=No+Image", width=200)
                    else:
                        st.image("https://via.placeholder.com/200x150/cccccc/666666?text=No+Image", width=200)
                    
                    st.markdown(f"**Vehicle #{vehicle['vehicle_id']}**")
                    st.caption(f"Seen {vehicle['count']} times")
                    st.caption(f"Last seen: {vehicle['last_seen']}")
                    
                    # Button to view timeline
                    if st.button(f"View Timeline", key=f"timeline_{vehicle['vehicle_id']}"):
                        st.session_state['selected_vehicle'] = vehicle['vehicle_id']
                        st.rerun()
            
            # Show selected vehicle info
            if 'selected_vehicle' in st.session_state:
                st.subheader(f"Selected Vehicle #{st.session_state['selected_vehicle']}")
                
                # Get detailed vehicle info
                try:
                    vehicle_response = requests.get(f"{API_BASE_URL}/vehicles/{st.session_state['selected_vehicle']}")
                    if vehicle_response.status_code == 200:
                        vehicle = vehicle_response.json()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**First Seen:** {vehicle['first_seen']}")
                            st.write(f"**Last Seen:** {vehicle['last_seen']}")
                            st.write(f"**Total Sightings:** {vehicle['count']}")
                        
                        with col2:
                            st.write(f"**Thumbnails:** {len(vehicle['thumbnail_paths'])}")
                            st.write(f"**Video Clips:** {len(vehicle['clip_paths'])}")
                        
                        # Show all thumbnails
                        if vehicle['thumbnail_paths']:
                            st.subheader("📸 All Sightings")
                            thumbnail_cols = st.columns(4)
                            for i, thumb_path in enumerate(vehicle['thumbnail_paths']):
                                with thumbnail_cols[i % 4]:
                                    if os.path.exists(thumb_path):
                                        st.image(thumb_path, width=150)
                                    else:
                                        st.image("https://via.placeholder.com/150x100/cccccc/666666?text=Missing", width=150)
                
                except Exception as e:
                    st.error(f"Error loading vehicle details: {e}")
        
        else:
            st.error("Failed to load vehicles")
    
    except Exception as e:
        st.error(f"Error connecting to API: {e}")


def show_timeline():
    """Show timeline for selected vehicle"""
    st.title("📊 Vehicle Timeline")
    
    if 'selected_vehicle' not in st.session_state:
        st.info("Please select a vehicle from the Gallery to view its timeline.")
        return
    
    vehicle_id = st.session_state['selected_vehicle']
    
    try:
        # Get vehicle timeline
        response = requests.get(f"{API_BASE_URL}/vehicles/{vehicle_id}")
        if response.status_code == 200:
            vehicle = response.json()
            
            st.subheader(f"Vehicle #{vehicle_id} Timeline")
            
            # Vehicle summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sightings", vehicle['count'])
            with col2:
                st.metric("First Seen", vehicle['first_seen'][:10])
            with col3:
                st.metric("Last Seen", vehicle['last_seen'][:10])
            
            # Timeline visualization
            st.subheader("📅 Detection Timeline")
            
            # Show sightings chronologically
            sightings = []
            for i, (thumb_path, clip_path) in enumerate(zip(vehicle['thumbnail_paths'], vehicle['clip_paths'])):
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
            st.error("Failed to load vehicle timeline")
    
    except Exception as e:
        st.error(f"Error loading timeline: {e}")


def show_playback():
    """Show video playback interface"""
    st.title("🎥 Video Playback")
    
    if 'selected_vehicle' not in st.session_state:
        st.info("Please select a vehicle from the Gallery to view its videos.")
        return
    
    vehicle_id = st.session_state['selected_vehicle']
    
    try:
        # Get vehicle clips
        response = requests.get(f"{API_BASE_URL}/vehicles/{vehicle_id}/clips")
        if response.status_code == 200:
            clips_data = response.json()
            
            st.subheader(f"Vehicle #{vehicle_id} Video Clips")
            
            clips = clips_data['clips']
            if not clips:
                st.info("No video clips available for this vehicle.")
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
                    value=f"vehicle_{vehicle_id}_merged.mp4",
                    help="Name for the merged video file"
                )
            
            with col2:
                if st.button("🔄 Generate Merged Video", type="primary"):
                    with st.spinner("Merging clips..."):
                        try:
                            merge_response = requests.post(
                                f"{API_BASE_URL}/vehicles/{vehicle_id}/merge",
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
            st.error("Failed to load vehicle clips")
    
    except Exception as e:
        st.error(f"Error loading playback: {e}")


def show_settings():
    """Show tracking settings and configuration"""
    st.title("⚙️ Settings")
    
    try:
        # Get current tracking status and config
        status_response = requests.get(f"{API_BASE_URL}/vehicles/tracking/status")
        config_response = requests.get(f"{API_BASE_URL}/vehicles/tracking/config")
        
        if status_response.status_code == 200 and config_response.status_code == 200:
            status = status_response.json()
            config = config_response.json()
            
            # Tracking control
            st.subheader("🎮 Tracking Control")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if status['is_running']:
                    if st.button("⏹️ Stop Tracking", type="secondary"):
                        try:
                            stop_response = requests.post(f"{API_BASE_URL}/vehicles/tracking/stop")
                            if stop_response.status_code == 200:
                                st.success("Tracking stopped")
                                st.rerun()
                            else:
                                st.error("Failed to stop tracking")
                        except Exception as e:
                            st.error(f"Error stopping tracking: {e}")
                else:
                    if st.button("▶️ Start Tracking", type="primary"):
                        try:
                            start_response = requests.post(f"{API_BASE_URL}/vehicles/tracking/start")
                            if start_response.status_code == 200:
                                st.success("Tracking started")
                                st.rerun()
                            else:
                                st.error("Failed to start tracking")
                        except Exception as e:
                            st.error(f"Error starting tracking: {e}")
            
            with col2:
                status_text = "🟢 Running" if status['is_running'] else "🔴 Stopped"
                st.metric("Current Status", status_text)
            
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
                        help="Threshold for matching vehicles"
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
                            f"{API_BASE_URL}/vehicles/tracking/config",
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
                storage_response = requests.get(f"{API_BASE_URL}/vehicles/storage/stats")
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
                                    f"{API_BASE_URL}/vehicles/storage/cleanup",
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


if __name__ == "__main__":
    main()
