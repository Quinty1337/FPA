import os
import cv2
import numpy as np
import pandas as pd
import tempfile
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDialogButtonBox, QPushButton


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
            dialog.setWindowTitle("Telemetry Overlay Settings")
            layout = QVBoxLayout(dialog)

            # Frame rate setting
            fr_layout = QHBoxLayout()
            fr_layout.addWidget(QLabel("Frame Rate (fps):"))
            fr_spin = QSpinBox()
            fr_spin.setRange(15, 120)
            fr_spin.setValue(30)
            fr_layout.addWidget(fr_spin)
            layout.addLayout(fr_layout)

            # Duration setting
            dur_layout = QHBoxLayout()
            dur_layout.addWidget(QLabel("Duration (seconds, 0 for all):"))
            dur_spin = QSpinBox()
            dur_spin.setRange(0, 3600)
            dur_spin.setValue(0)
            dur_layout.addWidget(dur_spin)
            layout.addLayout(dur_layout)

            # Buttons
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            # Get settings
            frame_rate = fr_spin.value()
            duration = dur_spin.value() if dur_spin.value() > 0 else None

            # Create the telemetry video
            self.console_output.append("Creating telemetry overlay video...")
            create_telemetry_video(
                self.processed_data,
                output_dir=overlay_dir,
                frame_rate=frame_rate,
                duration=duration,
                console_output=self.console_output
            )

        except Exception as e:
            self.console_output.append(f"Error: {str(e)}")
            import traceback
            self.console_output.append(traceback.format_exc())

    # Add the method to the parent class
    parent_class.generate_overlay_frames_clicked = generate_overlay_frames_clicked


def create_telemetry_video(data, output_dir, frame_rate=30, duration=None, console_output=None):
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