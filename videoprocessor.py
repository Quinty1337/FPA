import os
import cv2
import os
import numpy as np
import tempfile
import numpy as np
import pandas as pd
import tempfile
import ffmpeg
import subprocess
from datetime import timedelta
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDialogButtonBox,
    QPushButton, QFileDialog, QComboBox, QCheckBox, QDoubleSpinBox, QGridLayout,
    QTabWidget, QGroupBox, QWidget
)
from PyQt6.QtCore import Qt


def add_gui_integration(parent_class):
    """Add telemetry overlay video generation to the GUI"""

    def generate_overlay_frames_clicked(self):
        """Handle overlay generation button click"""
        # Check if flight data is loaded
        if self.performance_calculator is None:
            self.console_output.append("Error: Please load flight data first")
            return

        try:
            # Get project directory (where current file is)
            project_dir = os.path.dirname(os.path.abspath(self.current_file))

            # Create output directory in project folder
            overlay_dir = os.path.join(project_dir, "telemetry_overlay")
            os.makedirs(overlay_dir, exist_ok=True)
            self.console_output.append(f"Created output directory: {overlay_dir}")

            # Create settings dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Video Telemetry Generator")
            dialog.setMinimumWidth(600)
            layout = QVBoxLayout(dialog)
            
            # Create tab widget for different modes
            tab_widget = QTabWidget()
            layout.addWidget(tab_widget)
            
            # Tab 1: Generate synthetic video
            synthetic_tab = QWidget()
            synthetic_layout = QVBoxLayout(synthetic_tab)
            
            syn_group = QGroupBox("Synthetic Video Settings")
            syn_form = QGridLayout()
            
            # Frame rate setting
            syn_form.addWidget(QLabel("Frame Rate (fps):"), 0, 0)
            fr_spin = QSpinBox()
            fr_spin.setRange(15, 120)
            fr_spin.setValue(30)
            syn_form.addWidget(fr_spin, 0, 1)
            
            # Resolution setting
            syn_form.addWidget(QLabel("Resolution:"), 1, 0)
            res_combo = QComboBox()
            res_combo.addItems(["1280x720", "1920x1080", "3840x2160"])
            syn_form.addWidget(res_combo, 1, 1)
            
            # Duration setting
            syn_form.addWidget(QLabel("Duration (seconds, 0 for all):"), 2, 0)
            dur_spin = QSpinBox()
            dur_spin.setRange(0, 3600)
            dur_spin.setValue(0)
            syn_form.addWidget(dur_spin, 2, 1)
            
            syn_group.setLayout(syn_form)
            synthetic_layout.addWidget(syn_group)
            synthetic_layout.addStretch()
            
            # Tab 2: Overlay on existing video
            overlay_tab = QWidget()
            overlay_layout = QVBoxLayout(overlay_tab)
            
            # Video file selection
            video_group = QGroupBox("Video Source")
            video_form = QGridLayout()
            
            video_form.addWidget(QLabel("Video File:"), 0, 0)
            video_path_label = QLabel("No video selected")
            video_path_label.setWordWrap(True)
            video_form.addWidget(video_path_label, 0, 1)
            
            browse_button = QPushButton("Browse...")
            browse_button.clicked.connect(lambda: select_video_file(video_path_label))
            video_form.addWidget(browse_button, 0, 2)
            
            # Video sync settings
            video_form.addWidget(QLabel("Video Start Time (seconds in flight data):"), 1, 0)
            sync_spin = QDoubleSpinBox()
            sync_spin.setRange(0, 3600)
            sync_spin.setDecimals(1)
            sync_spin.setValue(0)
            video_form.addWidget(sync_spin, 1, 1)
            
            # Segment extraction
            video_form.addWidget(QLabel("Extract Segment:"), 2, 0)
            segment_check = QCheckBox("Enable")
            video_form.addWidget(segment_check, 2, 1)
            
            video_form.addWidget(QLabel("Start Time (seconds):"), 3, 0)
            segment_start = QDoubleSpinBox()
            segment_start.setRange(0, 3600)
            segment_start.setDecimals(1)
            segment_start.setValue(0)
            segment_start.setEnabled(False)
            video_form.addWidget(segment_start, 3, 1)
            
            video_form.addWidget(QLabel("Duration (seconds):"), 4, 0)
            segment_duration = QDoubleSpinBox()
            segment_duration.setRange(1, 3600)
            segment_duration.setDecimals(1)
            segment_duration.setValue(10)
            segment_duration.setEnabled(False)
            video_form.addWidget(segment_duration, 4, 1)
            
            # Connect segment checkbox to enable/disable fields
            segment_check.stateChanged.connect(lambda state: [
                segment_start.setEnabled(state == Qt.CheckState.Checked),
                segment_duration.setEnabled(state == Qt.CheckState.Checked)
            ])
            
            video_group.setLayout(video_form)
            overlay_layout.addWidget(video_group)
            overlay_layout.addStretch()
            
            # Add tabs to widget
            tab_widget.addTab(synthetic_tab, "Generate Synthetic Video")
            tab_widget.addTab(overlay_tab, "Overlay on Video")
            
            # Function to select video file
            def select_video_file(label):
                file_path, _ = QFileDialog.getOpenFileName(
                    dialog,
                    "Select Video File",
                    "",
                    "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
                )
                if file_path:
                    label.setText(file_path)
                    # Try to get video duration and set segment max values
                    try:
                        probe = ffmpeg.probe(file_path)
                        duration = float(probe['format']['duration'])
                        segment_start.setRange(0, duration)
                        segment_duration.setRange(1, duration)
                        segment_duration.setValue(min(10, duration))
                    except Exception as e:
                        self.console_output.append(f"Error reading video metadata: {str(e)}")
            
            # Buttons
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            
            # Process based on selected tab
            selected_tab = tab_widget.currentIndex()
            
            if selected_tab == 0:  # Synthetic video
                # Get settings
                frame_rate = fr_spin.value()
                duration = dur_spin.value() if dur_spin.value() > 0 else None
                resolution = res_combo.currentText().split('x')
                width, height = int(resolution[0]), int(resolution[1])
                
                # Create the telemetry video
                self.console_output.append("Creating synthetic telemetry video...")
                create_telemetry_video(
                    self.processed_data,
                    output_dir=overlay_dir,
                    frame_rate=frame_rate,
                    duration=duration,
                    width=width,
                    height=height,
                    console_output=self.console_output
                )
            else:  # Overlay on existing video
                video_path = video_path_label.text()
                if video_path == "No video selected":
                    self.console_output.append("Error: Please select a video file")
                    return
                
                # Get settings
                sync_time = sync_spin.value()
                
                # Check if segment extraction is enabled
                if segment_check.isChecked():
                    segment_start_time = segment_start.value()
                    segment_dur = segment_duration.value()
                else:
                    segment_start_time = None
                    segment_dur = None
                
                # Create overlay on existing video
                self.console_output.append("Creating telemetry overlay on video...")
                create_video_with_telemetry(
                    video_path,
                    self.processed_data,
                    output_dir=overlay_dir,
                    sync_offset=sync_time,
                    segment_start=segment_start_time,
                    segment_duration=segment_dur,
                    console_output=self.console_output
                )

        except Exception as e:
            self.console_output.append(f"Error: {str(e)}")
            import traceback
            self.console_output.append(traceback.format_exc())

    # Add the method to the parent class
    parent_class.generate_overlay_frames_clicked = generate_overlay_frames_clicked


def create_telemetry_video(data, output_dir, frame_rate=30, duration=None, width=1280, height=720, console_output=None):
    """
    Create a telemetry overlay video using OpenCV

    Args:
        data: Pandas DataFrame with flight data
        output_dir: Directory to save the video
        frame_rate: Video frame rate
        duration: Duration in seconds (None for full flight)
        console_output: Text widget to output messages (optional)
    """

    def log(message):
        """Helper to log messages to console and standard output"""
        print(message)
        if console_output:
            console_output.append(message)

    try:
        # Determine full duration if not specified
        if duration is None or duration == 0:
            max_time = data['second'].max()
            duration = max_time
            log(f"Using full flight duration: {duration:.1f} seconds")

        # Define video parameters
        width, height = 1280, 720
        output_file = os.path.join(output_dir, "telemetry_overlay.mp4")

        log(f"Creating {width}x{height} video at {frame_rate} fps")
        log(f"Output file: {output_file}")

        # Initialize video writer with mp4v codec (most compatible)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

        if not out.isOpened():
            log("Error: Failed to initialize video writer")
            return False

        # Calculate total frames
        total_frames = int(duration * frame_rate)
        log(f"Generating {total_frames} frames...")

        # Prepare HUD fonts and sizes
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255)  # White
        font_thickness = 2

        # Process each frame
        for frame_num in range(total_frames):
            # Current time in the flight
            current_time = frame_num / frame_rate

            # Show progress
            if frame_num % max(1, total_frames // 10) == 0:
                progress = (frame_num / total_frames) * 100
                log(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")

            # Find closest data point to current time
            closest_idx = (data['second'] - current_time).abs().idxmin()
            current_data = data.loc[closest_idx]

            # Create black background
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw semi-transparent background for telemetry info
            overlay = frame.copy()
            cv2.rectangle(overlay, (width - 400, 10), (width - 10, 210), (30, 30, 30), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Extract data values
            altitude = current_data.get('Approxaltitude[m]', 0)
            speed_ms = current_data.get('speed[m/s]', 0)
            speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
            vert_speed = current_data.get('vertical_speed', 0)

            # Add position if available
            lat = current_data.get('Latitude[deg]', 0)
            lon = current_data.get('Longitude[deg]', 0)

            # Format values with fixed width
            y_offset = 50
            text_spacing = 40

            # Add telemetry text
            cv2.putText(frame, f"Time: {current_time:.1f} s", (width - 380, y_offset),
                        font, font_scale, font_color, font_thickness)
            y_offset += text_spacing

            cv2.putText(frame, f"Altitude: {altitude:.1f} m", (width - 380, y_offset),
                        font, font_scale, font_color, font_thickness)
            y_offset += text_spacing

            cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (width - 380, y_offset),
                        font, font_scale, font_color, font_thickness)
            y_offset += text_spacing

            cv2.putText(frame, f"Vert Speed: {vert_speed:.1f} m/s", (width - 380, y_offset),
                        font, font_scale, font_color, font_thickness)
            y_offset += text_spacing

            cv2.putText(frame, f"Position: {lat:.6f}, {lon:.6f}", (width - 380, y_offset),
                        font, font_scale, font_color, font_thickness)

            # Write the frame
            out.write(frame)

        # Release video writer
        out.release()

        # Verify output
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            filesize_mb = os.path.getsize(output_file) / (1024 * 1024)
            log(f"Video created successfully: {output_file} ({filesize_mb:.2f} MB)")
            return True
        else:
            log("Error: Failed to create video file")
            return False

    except Exception as e:
        log(f"Error creating video: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return False
        
        
def create_video_with_telemetry(video_path, data, output_dir, sync_offset=0, segment_start=None, 
                           segment_duration=None, console_output=None):
    """
    Create a video with telemetry overlay using ffmpeg-python
    
    Args:
        video_path: Path to input video file
        data: Pandas DataFrame with flight data
        output_dir: Directory to save the output video
        sync_offset: Time offset in seconds to sync flight data with video (0 = video starts at beginning of flight data)
        segment_start: Start time in seconds to extract from video (None = start from beginning)
        segment_duration: Duration in seconds to extract (None = extract to end)
        console_output: Text widget to output messages (optional)
    """
    
    def log(message):
        """Helper to log messages to console and standard output"""
        print(message)
        if console_output:
            console_output.append(message)
    
    try:
        # Get video metadata
        log(f"Getting metadata for video: {video_path}")
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        
        # Get video properties
        width = int(video_info['width'])
        height = int(video_info['height'])
        
        # Determine frame rate
        fps_parts = video_info.get('avg_frame_rate', '30/1').split('/')
        frame_rate = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else 30
        
        # Get video duration
        video_duration = float(probe['format']['duration'])
        log(f"Video properties: {width}x{height}, {frame_rate} fps, {video_duration:.2f} seconds")
        
        # Set up segment extraction if needed
        input_args = {}
        if segment_start is not None:
            input_args['ss'] = str(segment_start)
            
            if segment_duration is not None:
                input_args['t'] = str(segment_duration)
                log(f"Extracting segment: {segment_start}s to {segment_start + segment_duration}s")
            else:
                log(f"Extracting segment: {segment_start}s to end")
        
        # Define output file
        output_filename = "telemetry_overlay_video.mp4"
        if segment_start is not None:
            # Include segment info in filename
            segment_end = segment_start + segment_duration if segment_duration else "end"
            output_filename = f"telemetry_segment_{segment_start}s_to_{segment_end}s.mp4"
        
        output_file = os.path.join(output_dir, output_filename)
        log(f"Output file: {output_file}")
        
        # Create temporary directory for overlay frames
        with tempfile.TemporaryDirectory() as temp_dir:
            log(f"Using temporary directory: {temp_dir}")
            
            # Create a transparent overlay with just the telemetry text
            overlay_file = os.path.join(temp_dir, "overlay.png")
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_color = (255, 255, 255)  # White
            font_thickness = 2
            
            # Create telemetry frames
            log("Generating telemetry overlay frames...")
            
            # Get actual duration to process
            process_duration = segment_duration if segment_duration else video_duration
            if segment_start and not segment_duration:
                process_duration = video_duration - segment_start
                
            # Calculate total frames
            total_frames = int(process_duration * frame_rate)
            
            # Generate overlay frames
            for frame_num in range(total_frames):
                # Current time in the video
                video_time = frame_num / frame_rate
                
                # Convert to flight data time using sync offset
                # If segment_start is specified, add it to get correct video position
                if segment_start is not None:
                    flight_time = sync_offset + segment_start + video_time
                else:
                    flight_time = sync_offset + video_time
                
                # Show progress
                if frame_num % max(1, total_frames // 10) == 0:
                    progress = (frame_num / total_frames) * 100
                    log(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")
                
                # Find closest data point to current time in flight data
                if flight_time <= data['second'].max() and flight_time >= data['second'].min():
                    closest_idx = (data['second'] - flight_time).abs().idxmin()
                    current_data = data.loc[closest_idx]
                    
                    # Create transparent overlay
                    overlay = np.zeros((height, width, 4), dtype=np.uint8)
                    
                    # Create semi-transparent background for telemetry
                    cv2.rectangle(overlay, (width - 400, 10), (width - 10, 210), (30, 30, 30, 180), -1)
                    
                    # Extract data values
                    altitude = current_data.get('Approxaltitude[m]', 0)
                    speed_ms = current_data.get('speed[m/s]', 0)
                    speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
                    vert_speed = current_data.get('vertical_speed', 0)
                    
                    # Add position if available
                    lat = current_data.get('Latitude[deg]', 0)
                    lon = current_data.get('Longitude[deg]', 0)
                    
                    # Format values with fixed width
                    y_offset = 50
                    text_spacing = 40
                    
                    # Add telemetry text
                    cv2.putText(overlay, f"Time: {flight_time:.1f} s", (width - 380, y_offset),
                                font, font_scale, font_color, font_thickness)
                    y_offset += text_spacing
                    
                    cv2.putText(overlay, f"Altitude: {altitude:.1f} m", (width - 380, y_offset),
                                font, font_scale, font_color, font_thickness)
                    y_offset += text_spacing
                    
                    cv2.putText(overlay, f"Speed: {speed_kmh:.1f} km/h", (width - 380, y_offset),
                                font, font_scale, font_color, font_thickness)
                    y_offset += text_spacing
                    
                    cv2.putText(overlay, f"Vert Speed: {vert_speed:.1f} m/s", (width - 380, y_offset),
                                font, font_scale, font_color, font_thickness)
                    y_offset += text_spacing
                    
                    cv2.putText(overlay, f"Position: {lat:.6f}, {lon:.6f}", (width - 380, y_offset),
                                font, font_scale, font_color, font_thickness)
                else:
                    # If outside flight data range, just show a minimal overlay
                    overlay = np.zeros((height, width, 4), dtype=np.uint8)
                    cv2.rectangle(overlay, (width - 400, 10), (width - 10, 90), (30, 30, 30, 180), -1)
                    cv2.putText(overlay, f"Video time: {video_time:.1f} s", (width - 380, 50),
                                font, font_scale, font_color, font_thickness)
                    cv2.putText(overlay, "No flight data available", (width - 380, 90),
                                font, font_scale, font_color, font_thickness)
                
                # Save overlay frame
                overlay_path = os.path.join(temp_dir, f"overlay_{frame_num:06d}.png")
                cv2.imwrite(overlay_path, overlay)
            
            # Use ffmpeg to combine video with overlay
            log("Combining video with telemetry overlay using ffmpeg...")
            
            # Create ffmpeg input for video
            video_input = ffmpeg.input(video_path, **input_args)
            
            # Create input for overlay frames
            overlay_pattern = os.path.join(temp_dir, "overlay_%06d.png")
            overlay_input = ffmpeg.input(overlay_pattern, framerate=frame_rate)
            
            # Combine video with overlay
            output = (
                ffmpeg
                .filter([video_input, overlay_input], 'overlay', 0, 0)
                .output(output_file, 
                       vcodec='libx264', 
                       pix_fmt='yuv420p',
                       preset='medium', 
                       crf=23)
                .overwrite_output()
                .global_args('-loglevel', 'error')
            )
            
            # Run ffmpeg command
            log("Processing video with ffmpeg...")
            output.run(capture_stdout=True, capture_stderr=True)
            
            # Verify output
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                filesize_mb = os.path.getsize(output_file) / (1024 * 1024)
                log(f"Video created successfully: {output_file} ({filesize_mb:.2f} MB)")
                return True
            else:
                log("Error: Failed to create video file")
                return False
    
    except Exception as e:
        log(f"Error creating video with telemetry: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return False