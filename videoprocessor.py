import cv2
import numpy as np
import pandas as pd
import os
import time
import traceback
import shutil
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, \
    QPushButton, QGroupBox, QApplication
from PyQt6.QtCore import Qt

# Utility class for progress reporting
class ProgressLogger:
    """Helper class for logging progress with timestamps without stopping execution"""
    
    def __init__(self, prefix=""):
        """Initialize with optional prefix for all messages"""
        self.prefix = prefix
        self.start_time = time.time()
        self.last_time = self.start_time
    
    def log(self, message, show_elapsed=False):
        """Log a message with optional elapsed time"""
        now = time.time()
        if show_elapsed:
            elapsed = now - self.start_time
            time_str = f"[{elapsed:.1f}s] "
        else:
            time_str = ""
        
        if self.prefix:
            print(f"{self.prefix} {time_str}{message}")
        else:
            print(f"{time_str}{message}")
        
        self.last_time = now
        return now
    
    def progress(self, current, total, message="", min_interval=1.0):
        """
        Log progress message only if sufficient time has passed since last update
        to avoid overwhelming the console
        """
        now = time.time()
        if now - self.last_time >= min_interval or current >= total:
            percent = (current / total) * 100 if total > 0 else 0
            elapsed = now - self.start_time
            eta = (elapsed / current) * (total - current) if current > 0 else 0
            
            if message:
                print(f"{self.prefix} {message}: {percent:.1f}% ({current}/{total}) - Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
            else:
                print(f"{self.prefix} Progress: {percent:.1f}% ({current}/{total}) - Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
            
            self.last_time = now
        return now


class VideoProcessor:
    """Class for processing video and overlaying flight telemetry data"""
    
    def __init__(self, video_path=None, flight_data=None):
        """Initialize with video path and flight data DataFrame"""
        self.video_path = video_path
        self.flight_data = flight_data
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.duration = 0
        self.frame_width = 0
        self.frame_height = 0
        self.output_dir = None
        
        # Font configuration
        try:
            font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'arial.ttf')
            if os.path.exists(font_path):
                self.font_path = font_path
            else:
                # Try to use system fonts as fallback
                from matplotlib import rcParams
                self.font_path = rcParams['font.family']
                print(f"Font not found at {font_path}, using system font: {self.font_path}")
        except Exception as e:
            print(f"Error finding fonts: {e}. Falling back to default font.")
            self.font_path = None
            
        # Default color scheme (green for HUD elements)
        self.hud_color = (0, 255, 0)  # Green in BGR
        self.warning_color = (0, 0, 255)  # Red in BGR
        
    def load_video(self):
        """Load video file and extract properties"""
        if not self.video_path or not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {self.video_path}")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Validate video properties to avoid issues later
        if self.fps <= 0 or self.frame_count <= 0 or self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError(f"Invalid video properties detected: FPS={self.fps}, frames={self.frame_count}, "
                            f"dimensions={self.frame_width}x{self.frame_height}. "
                            f"The video file may be corrupted or unsupported.")
        
        print(f"Loaded video: {os.path.basename(self.video_path)}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps:.2f}")
        print(f"Duration: {self.duration:.2f} seconds ({self.frame_count} frames)")
        
        return True
        
    def sync_video_with_data(self, time_offset=0):
        """
        Synchronize video with flight data using a time offset.
        
        Args:
            time_offset: Offset in seconds to align video with telemetry data
                         (positive if video starts after telemetry, negative if before)
        """
        if self.flight_data is None:
            raise ValueError("Flight data not loaded. Load flight data first.")
            
        # Create a mapping from video frame number to flight data index
        # Using video time (in seconds) + offset to find the corresponding data points
        self.frame_to_data_map = {}
        
        # Get the start time in seconds from the flight data
        min_time = self.flight_data['second'].min()
        
        for frame_num in range(self.frame_count):
            # Calculate the time in seconds for this frame
            frame_time = frame_num / self.fps
            
            # Adjust with offset and find the closest time in flight data
            adjusted_time = frame_time + time_offset + min_time
            
            # Find the closest time in the flight data
            closest_idx = (self.flight_data['second'] - adjusted_time).abs().idxmin()
            self.frame_to_data_map[frame_num] = closest_idx
            
        print(f"Synchronized video with flight data (offset: {time_offset} seconds)")
        return True
        
    def get_data_for_frame(self, frame_num):
        """Get flight data corresponding to a specific video frame"""
        if frame_num not in self.frame_to_data_map:
            return None
            
        data_idx = self.frame_to_data_map[frame_num]
        return self.flight_data.loc[data_idx]
        
    def create_hud_overlay(self, frame, frame_data):
        """
        Create a HUD (Heads-Up Display) overlay with flight data
        
        Args:
            frame: Video frame as numpy array
            frame_data: Pandas Series with flight data for this frame
        
        Returns:
            Frame with HUD overlay
        """
        # Convert OpenCV BGR frame to PIL Image for easier text handling
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Try to use custom font if available
        try:
            # Use different font sizes for different elements
            large_font_size = max(16, int(self.frame_height / 30))
            medium_font_size = max(14, int(self.frame_height / 40))
            small_font_size = max(12, int(self.frame_height / 50))
            
            if self.font_path:
                large_font = ImageFont.truetype(self.font_path, large_font_size)
                medium_font = ImageFont.truetype(self.font_path, medium_font_size)
                small_font = ImageFont.truetype(self.font_path, small_font_size)
            else:
                # Fall back to default font
                large_font = ImageFont.load_default()
                medium_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        except Exception as e:
            print(f"Error loading fonts: {e}")
            large_font = ImageFont.load_default()
            medium_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Convert HUD color from BGR to RGB for PIL
        hud_color_rgb = (self.hud_color[2], self.hud_color[1], self.hud_color[0])
        warning_color_rgb = (self.warning_color[2], self.warning_color[1], self.warning_color[0])
        
        # Extract the required data
        speed = frame_data.get('speed[m/s]', 0)
        altitude = frame_data.get('Approxaltitude[m]', 0)
        vertical_speed = frame_data.get('vertical_speed', 0)
        
        # Display speed directly in m/s without conversion to km/h
        
        # Format speed and altitude with proper units
        speed_text = f"SPD: {speed:.1f} m/s"
        altitude_text = f"ALT: {altitude:.1f} m"
        vspeed_text = f"V/S: {vertical_speed:.1f} m/s"
        
        # Define positions for the HUD elements
        # Speed on the left side
        speed_pos = (20, 30)
        # Altitude on the right side
        altitude_pos = (self.frame_width - 150, 30)
        # Vertical speed below altitude
        vspeed_pos = (self.frame_width - 150, 70)
        
        # Draw the HUD elements
        draw.text(speed_pos, speed_text, fill=hud_color_rgb, font=large_font)
        draw.text(altitude_pos, altitude_text, fill=hud_color_rgb, font=large_font)
        draw.text(vspeed_pos, vspeed_text, fill=hud_color_rgb, font=medium_font)
        
        # Add time counter at the bottom
        time_seconds = frame_data.get('second', 0)
        # Handle various numeric types including numpy types
        if hasattr(time_seconds, 'item'):
            time_seconds = time_seconds.item()  # Convert numpy types to native Python types
        else:
            try:
                time_seconds = int(time_seconds)  # Try to convert to int
            except (TypeError, ValueError):
                time_seconds = 0  # Default if conversion fails
                
        time_formatted = str(timedelta(seconds=time_seconds))
        if "." in time_formatted:
            time_formatted = time_formatted.split(".")[0]  # Remove microseconds
        time_pos = (20, self.frame_height - 40)
        draw.text(time_pos, f"TIME: {time_formatted}", fill=hud_color_rgb, font=medium_font)
        
        # Draw artificial horizon line (simplified)
        horizon_length = int(self.frame_width * 0.4)
        horizon_center_x = self.frame_width // 2
        horizon_center_y = self.frame_height // 2
        
        # Draw central crosshair/reticle
        crosshair_size = 20
        line_thickness = 2
        
        # Horizontal line
        draw.line(
            (horizon_center_x - crosshair_size, horizon_center_y, 
             horizon_center_x + crosshair_size, horizon_center_y),
            fill=hud_color_rgb, width=line_thickness
        )
        
        # Vertical line
        draw.line(
            (horizon_center_x, horizon_center_y - crosshair_size, 
             horizon_center_x, horizon_center_y + crosshair_size),
            fill=hud_color_rgb, width=line_thickness
        )
        
        # Draw altitude indicator bars on right side (like a ladder)
        ladder_height = 200
        ladder_right = self.frame_width - 20
        ladder_center_y = self.frame_height // 2
        
        # Draw the ladder ticks
        for i in range(-2, 3):
            tick_y = ladder_center_y - i * (ladder_height // 5)
            tick_alt = altitude + i * 10  # 10m increments
            
            # Draw tick line
            tick_length = 15 if i == 0 else 10
            draw.line(
                (ladder_right - tick_length, tick_y, ladder_right, tick_y),
                fill=hud_color_rgb, width=line_thickness
            )
            
            # Only label every other tick to avoid clutter
            if i % 1 == 0:
                # Draw altitude label
                tick_label = f"{tick_alt:.0f}"
                draw.text(
                    (ladder_right - tick_length - 40, tick_y - 10),
                    tick_label, fill=hud_color_rgb, font=small_font
                )
        
        # Convert back to OpenCV format
        result_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return result_frame
    
    def process_video(self, output_dir=None, time_offset=0, start_frame=0, end_frame=None):
        """
        Process video with flight data overlay and save frames
        
        Args:
            output_dir: Directory to save output frames (default: project_folder/processed_frames)
            time_offset: Time offset in seconds to sync video with flight data
            start_frame: First frame to process (default: 0)
            end_frame: Last frame to process (default: all frames)
        """
        # Create a progress logger for this operation
        logger = ProgressLogger(prefix="[Frame Processing]")
        logger.log("Starting video processing...")
        
        try:
            if not self.cap:
                logger.log("Loading video file...")
                if not self.load_video():
                    logger.log("Failed to load video file", show_elapsed=True)
                    return False
                    
            if self.flight_data is None:
                raise ValueError("Flight data not loaded. Load flight data first.")
            
            # If no output directory specified, use project folder
            if output_dir is None:
                # Use current directory (project root) for frames
                project_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(project_dir, "processed_frames")
                
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
            
            # Synchronize video with flight data
            logger.log(f"Synchronizing video with flight data (offset: {time_offset}s)...")
            self.sync_video_with_data(time_offset)
            
            # Set frame range
            if end_frame is None or end_frame > self.frame_count:
                end_frame = self.frame_count
                
            # Validate frame range
            if start_frame >= end_frame:
                raise ValueError(f"Invalid frame range: start ({start_frame}) must be less than end ({end_frame})")
                
            # Set video position to start frame
            if start_frame > 0:
                logger.log(f"Setting start position to frame {start_frame}...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
            total_frames = end_frame - start_frame
            logger.log(f"Processing {total_frames} frames from {start_frame} to {end_frame-1}...")
            
            # Process frames
            processed_count = 0
            error_count = 0
            max_errors = min(50, total_frames // 10)  # Allow up to 10% errors or 50, whichever is smaller
            skipped_frames = []
            
            for frame_num in range(start_frame, end_frame):
                try:
                    # Read frame
                    ret, frame = self.cap.read()
                    if not ret:
                        error_count += 1
                        skipped_frames.append(frame_num)
                        if error_count <= 5:  # Only show first few errors to avoid spam
                            logger.log(f"Failed to read frame {frame_num}")
                        if error_count > max_errors:
                            logger.log(f"Too many frame reading errors ({error_count}). Stopping processing.")
                            break
                        continue
                        
                    # Get flight data for this frame
                    frame_data = self.get_data_for_frame(frame_num)
                    if frame_data is None:
                        if processed_count < 5:  # Only show warning for first few frames
                            logger.log(f"Warning: No flight data for frame {frame_num}, using empty data")
                        # Create empty data if none exists to avoid errors
                        frame_data = pd.Series({
                            'second': 0,
                            'speed[m/s]': 0,
                            'Approxaltitude[m]': 0,
                            'vertical_speed': 0
                        })
                        
                    # Create overlay
                    processed_frame = self.create_hud_overlay(frame, frame_data)
                    
                    # Save frame
                    output_file = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
                    cv2.imwrite(output_file, processed_frame)
                    
                    # Update progress
                    processed_count += 1
                    logger.progress(processed_count, total_frames)
                    
                except Exception as e:
                    error_count += 1
                    skipped_frames.append(frame_num)
                    if error_count <= 5:  # Only show first few errors to avoid spam
                        logger.log(f"Error processing frame {frame_num}: {str(e)}")
                    if error_count > max_errors:
                        logger.log(f"Too many processing errors ({error_count}). Stopping processing.")
                        break
            
            # Final progress update
            success_rate = (processed_count / total_frames) * 100 if total_frames > 0 else 0
            logger.log(f"Processed {processed_count} of {total_frames} frames ({success_rate:.1f}%). "
                      f"Output saved to {output_dir}", show_elapsed=True)
            
            if error_count > 0:
                logger.log(f"Warning: Encountered {error_count} errors during processing.")
                if len(skipped_frames) > 0:
                    skip_ranges = []
                    start = skipped_frames[0]
                    prev = start
                    
                    for i in range(1, len(skipped_frames)):
                        if skipped_frames[i] != prev + 1:
                            if start == prev:
                                skip_ranges.append(f"{start}")
                            else:
                                skip_ranges.append(f"{start}-{prev}")
                            start = skipped_frames[i]
                        prev = skipped_frames[i]
                    
                    if start == prev:
                        skip_ranges.append(f"{start}")
                    else:
                        skip_ranges.append(f"{start}-{prev}")
                        
                    skip_str = ", ".join(skip_ranges[:5])
                    if len(skip_ranges) > 5:
                        skip_str += f", ... and {len(skip_ranges) - 5} more ranges"
                    
                    logger.log(f"Skipped frames: {skip_str}")
                
            return processed_count > 0  # Return True if at least some frames were processed
            
        except Exception as e:
            logger.log(f"Error during video processing: {str(e)}")
            logger.log(traceback.format_exc())
            return False
        finally:
            # Ensure video capture is released even if an exception occurs
            if self.cap and hasattr(self.cap, 'isOpened') and self.cap.isOpened():
                logger.log("Releasing video capture resources")
                self.cap.release()
        
    def create_output_video(self, output_path=None, frame_rate=None, clean_frames=True):
        """
        Create video from processed frames
        
        Args:
            output_path: Path to save output video (default: same as input with _overlay suffix)
            frame_rate: Frame rate of output video (default: same as input)
            clean_frames: Whether to clean up frame files after successful video creation (default: True)
            :type output_path: object
        """
        logger = ProgressLogger(prefix="[Video Creation]")
        logger.log("Starting output video creation...")
        video_writer = None
        
        try:
            if self.output_dir is None or not os.path.exists(self.output_dir):
                raise ValueError("No processed frames found. Process video first.")
                
            # Set default output path if not provided
            if output_path is None:
                # Default to saving in project directory
                project_dir = os.path.dirname(os.path.abspath(__file__))
                
                if self.video_path:
                    filename = os.path.splitext(os.path.basename(self.video_path))[0]
                    output_path = os.path.join(project_dir, f"{filename}_overlay.mp4")
                else:
                    output_path = os.path.join(project_dir, "output_overlay.mp4")
                    
            # Ensure we're not trying to overwrite the original video
            if self.video_path and os.path.abspath(output_path) == os.path.abspath(self.video_path):
                logger.log("Warning: Attempting to overwrite original video. Changing output path.")
                project_dir = os.path.dirname(os.path.abspath(__file__))
                filename = os.path.splitext(os.path.basename(self.video_path))[0]
                output_path = os.path.join(project_dir, f"{filename}_overlay_new.mp4")
                
            # Set default frame rate if not provided
            if frame_rate is None:
                frame_rate = self.fps
                
            # Get list of frame files
            frame_files = sorted([f for f in os.listdir(self.output_dir) if f.startswith("frame_") and f.endswith(".jpg")])
            if not frame_files:
                raise ValueError("No processed frames found in output directory.")
                
            # Read first frame to get dimensions
            first_frame = cv2.imread(os.path.join(self.output_dir, frame_files[0]))
            height, width, _ = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264
            video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
            
            # Add frames to video
            for frame_file in frame_files:
                frame_path = os.path.join(self.output_dir, frame_file)
                frame = cv2.imread(frame_path)
                video_writer.write(frame)
                
            video_writer.release()
            logger.log(f"Output video saved to {output_path}")
            
            # Clean up frame files if requested
            if clean_frames:
                self._clean_frame_files(logger)
        except Exception as e:
                if logger:
                    logger.log(f"Error : {str(e)}")
        return output_path

    def _clean_frame_files(self, logger=None):
        """
        Clean up frame files to save disk space after video creation
        
        Args:
            logger: Optional logger for progress messages
        """
        if logger:
            logger.log("Cleaning up processed frame files...")
        
        if not self.output_dir or not os.path.exists(self.output_dir):
            if logger:
                logger.log("No frame directory to clean")
            return
        
        try:
            # Count frame files before deletion
            frame_files = [f for f in os.listdir(self.output_dir)
                          if f.startswith("frame_") and f.endswith(".jpg")]
            frame_count = len(frame_files)
            
            if frame_count == 0:
                if logger:
                    logger.log("No frame files to clean")
                return
            
            # Process in batches to avoid memory issues with very large directories
            batch_size = 1000
            batches = (frame_count + batch_size - 1) // batch_size  # Ceiling division
                
            if logger and batches > 1:
                logger.log(f"Deleting {frame_count} files in {batches} batches")
                
            deleted_count = 0
            for batch_idx in range(batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, frame_count)
                    
                for file_name in frame_files[start_idx:end_idx]:
                    file_path = os.path.join(self.output_dir, file_name)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        if logger:
                            logger.log(f"Error deleting {file_name}: {e}")
                            
                if logger and batches > 1:
                    logger.log(f"Batch {batch_idx+1}/{batches} completed: {deleted_count}/{frame_count} files deleted")
            
            if logger:
                logger.log(f"Successfully removed {deleted_count} of {frame_count} frame files")
            
        except Exception as e:
            if logger:
                logger.log(f"Error cleaning frame files: {str(e)}")
            

class VideoOptionsDialog(QDialog):
    """Dialog for configuring video processing options"""
    
    def __init__(self, parent=None, video_duration=0, frame_count=0):
        super().__init__(parent)
        self.setWindowTitle("Video Processing Options")
        self.setMinimumWidth(450)
        
        # Store video properties
        self.video_duration = video_duration
        self.frame_count = frame_count
        self.fps = frame_count / video_duration if video_duration > 0 else 30
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Video info at the top
        info_group = QGroupBox("Video Information")
        info_layout = QVBoxLayout()
        
        if video_duration > 0 and frame_count > 0:
            info_layout.addWidget(QLabel(f"Duration: {video_duration:.2f} seconds"))
            info_layout.addWidget(QLabel(f"Frame Count: {frame_count}"))
            info_layout.addWidget(QLabel(f"Frame Rate: {self.fps:.2f} fps"))
        else:
            info_layout.addWidget(QLabel("Video information not available"))
            
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Frame range group
        frame_group = QGroupBox("Video Section to Render")
        frame_layout = QVBoxLayout()
        
        # Start frame with time display
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start Frame:"))
        self.start_frame = QSpinBox()
        self.start_frame.setRange(0, max(0, frame_count - 1))
        self.start_frame.setValue(0)
        self.start_frame.setMinimumWidth(100)
        start_layout.addWidget(self.start_frame)
        
        self.start_time_label = QLabel("(0.0s)")
        start_layout.addWidget(self.start_time_label)
        start_layout.addStretch()
        frame_layout.addLayout(start_layout)
        
        # End frame with time display
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End Frame:"))
        self.end_frame = QSpinBox()
        self.end_frame.setRange(1, frame_count)
        self.end_frame.setValue(frame_count)
        self.end_frame.setMinimumWidth(100)
        end_layout.addWidget(self.end_frame)
        
        self.end_time_label = QLabel(f"({video_duration:.1f}s)")
        end_layout.addWidget(self.end_time_label)
        end_layout.addStretch()
        frame_layout.addLayout(end_layout)
        
        # Selection duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Selected Duration:"))
        self.duration_label = QLabel(f"{video_duration:.1f} seconds")
        duration_layout.addWidget(self.duration_label)
        duration_layout.addStretch()
        frame_layout.addLayout(duration_layout)
        
        # Quick selection buttons for common scenarios
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("Quick Select:"))
        
        full_button = QPushButton("Full Video")
        full_button.clicked.connect(self.select_full_video)
        quick_layout.addWidget(full_button)
        
        if frame_count > 0:
            first_min = QPushButton("First Minute")
            first_min.clicked.connect(lambda: self.select_time_range(0, 60))
            quick_layout.addWidget(first_min)
            
            middle_button = QPushButton("Middle Section")
            middle_button.clicked.connect(self.select_middle_section)
            quick_layout.addWidget(middle_button)
        
        quick_layout.addStretch()
        frame_layout.addLayout(quick_layout)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # Synchronization group
        sync_group = QGroupBox("Data Synchronization")
        sync_layout = QVBoxLayout()
        
        # Offset explanation
        sync_layout.addWidget(QLabel("Adjust time offset to synchronize video with telemetry data:"))
        
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Time Offset (seconds):"))
        self.time_offset = QDoubleSpinBox()
        self.time_offset.setRange(-3600, 3600)  # ±1 hour should be enough
        self.time_offset.setDecimals(1)
        self.time_offset.setSingleStep(0.1)
        self.time_offset.setValue(0.0)
        self.time_offset.setMinimumWidth(100)
        offset_layout.addWidget(self.time_offset)
        
        offset_help = QLabel("(+ if video starts after data, - if before)")
        offset_help.setStyleSheet("color: gray;")
        offset_layout.addWidget(offset_help)
        offset_layout.addStretch()
        sync_layout.addLayout(offset_layout)
        
        # Quick offset buttons
        quick_offset_layout = QHBoxLayout()
        quick_offset_layout.addWidget(QLabel("Quick Adjust:"))
        
        minus_5 = QPushButton("-5s")
        minus_5.clicked.connect(lambda: self.adjust_offset(-5))
        quick_offset_layout.addWidget(minus_5)
        
        minus_1 = QPushButton("-1s")
        minus_1.clicked.connect(lambda: self.adjust_offset(-1))
        quick_offset_layout.addWidget(minus_1)
        
        plus_1 = QPushButton("+1s")
        plus_1.clicked.connect(lambda: self.adjust_offset(1))
        quick_offset_layout.addWidget(plus_1)
        
        plus_5 = QPushButton("+5s")
        plus_5.clicked.connect(lambda: self.adjust_offset(5))
        quick_offset_layout.addWidget(plus_5)
        
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: self.time_offset.setValue(0.0))
        quick_offset_layout.addWidget(reset)
        
        quick_offset_layout.addStretch()
        sync_layout.addLayout(quick_offset_layout)
        
        sync_group.setLayout(sync_layout)
        layout.addWidget(sync_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button = QPushButton("Process Video")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setDefault(True)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        layout.addLayout(button_layout)
        
        # Connect signals to update time labels
        self.start_frame.valueChanged.connect(self.update_time_labels)
        self.end_frame.valueChanged.connect(self.update_time_labels)
        
        # Initial update
        self.update_time_labels()
    
    def update_time_labels(self):
        """Update all time-related labels based on selected frames"""
        if self.fps > 0:
            start_time = self.start_frame.value() / self.fps
            end_time = self.end_frame.value() / self.fps
            duration = end_time - start_time
            
            self.start_time_label.setText(f"({start_time:.1f}s)")
            self.end_time_label.setText(f"({end_time:.1f}s)")
            self.duration_label.setText(f"{duration:.1f} seconds")
    
    def select_full_video(self):
        """Select the entire video"""
        self.start_frame.setValue(0)
        self.end_frame.setValue(self.frame_count)
    
    def select_time_range(self, start_seconds, end_seconds):
        """Select a range based on time in seconds"""
        if self.fps > 0:
            start_frame = max(0, min(int(start_seconds * self.fps), self.frame_count - 1))
            end_frame = max(1, min(int(end_seconds * self.fps), self.frame_count))
            
            self.start_frame.setValue(start_frame)
            self.end_frame.setValue(end_frame)
    
    def select_middle_section(self):
        """Select the middle third of the video"""
        if self.frame_count > 0:
            third = self.frame_count // 3
            self.start_frame.setValue(third)
            self.end_frame.setValue(third * 2)
    
    def adjust_offset(self, delta):
        """Adjust the time offset by the specified amount"""
        current = self.time_offset.value()
        self.time_offset.setValue(current + delta)
    
    def select_full_video(self):
        """Select the entire video"""
        self.start_frame.setValue(0)
        self.end_frame.setValue(self.frame_count)
    
    def select_time_range(self, start_seconds, end_seconds):
        """Select a range based on time in seconds"""
        if self.fps > 0:
            start_frame = max(0, min(int(start_seconds * self.fps), self.frame_count - 1))
            end_frame = max(1, min(int(end_seconds * self.fps), self.frame_count))
            
            self.start_frame.setValue(start_frame)
            self.end_frame.setValue(end_frame)
    
    def select_middle_section(self):
        """Select the middle third of the video"""
        if self.frame_count > 0:
            third = self.frame_count // 3
            self.start_frame.setValue(third)
            self.end_frame.setValue(third * 2)
    
    def adjust_offset(self, delta):
        """Adjust the time offset by the specified amount"""
        current = self.time_offset.value()
        self.time_offset.setValue(current + delta)
    
    def select_full_video(self):
        """Select the entire video"""
        self.start_frame.setValue(0)
        self.end_frame.setValue(self.frame_count)
    
    def select_time_range(self, start_seconds, end_seconds):
        """Select a range based on time in seconds"""
        if self.fps > 0:
            start_frame = max(0, min(int(start_seconds * self.fps), self.frame_count - 1))
            end_frame = max(1, min(int(end_seconds * self.fps), self.frame_count))
            
            self.start_frame.setValue(start_frame)
            self.end_frame.setValue(end_frame)
    
    def select_middle_section(self):
        """Select the middle third of the video"""
        if self.frame_count > 0:
            third = self.frame_count // 3
            self.start_frame.setValue(third)
            self.end_frame.setValue(third * 2)
    
    def adjust_offset(self, delta):
        """Adjust the time offset by the specified amount"""
        current = self.time_offset.value()
        self.time_offset.setValue(current + delta)
    
    def get_options(self):
        """Get the selected options as a dictionary"""
        return {
            'start_frame': self.start_frame.value(),
            'end_frame': self.end_frame.value(),
            'time_offset': self.time_offset.value()
        }


def add_gui_integration(gui_class):
    """
    Add video processing functionality to GUI class
    
    Args:
        gui_class: GUI class to add functionality to (must be a class, not an instance)
    """
    def generate_overlay_frames_clicked(self):
        """Method to handle 'Generate Telemetry Overlay' button click"""
        if not hasattr(self, 'performance_calculator') or self.performance_calculator is None:
            self.console_output.append("Please load flight data first")
            return
            
        # Ask user for video file
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if not video_path:
            return
            
        # Update UI to show processing status
        self.export_status.setText("Preparing video processing...")
        self.export_status.setStyleSheet("color: blue")
        
        # Make sure console tab is visible
        if hasattr(self, 'main_tabs'):
            for i in range(self.main_tabs.count()):
                if self.main_tabs.tabText(i) == "Console":
                    self.main_tabs.setCurrentIndex(i)
                    break
        
        try:
            self.console_output.append(f"Selected video: {os.path.basename(video_path)}")
            
            # Create video processor
            processor = VideoProcessor(
                video_path=video_path,
                flight_data=self.performance_calculator.df
            )
            
            # Update status
            self.export_status.setText("Loading video file...")
            self.export_status.setStyleSheet("color: blue")
            
            # Load video to get properties
            processor.load_video()
            
            # Show options dialog
            options_dialog = VideoOptionsDialog(
                parent=self,
                video_duration=processor.duration,
                frame_count=processor.frame_count
            )
            
            if not options_dialog.exec():
                self.console_output.append("Video processing cancelled by user")
                self.export_status.setText("Video processing cancelled")
                self.export_status.setStyleSheet("color: gray")
                return
                
            # Get selected options
            options = options_dialog.get_options()
            
            # Create output directory
            output_dir = os.path.join(os.path.dirname(video_path), "processed_frames")
            
            # Calculate expected duration and display
            total_frames = options['end_frame'] - options['start_frame']
            estimated_time = total_frames / processor.fps  # seconds per frame
            
            # Process video with selected options
            self.console_output.append("=" * 50)
            self.console_output.append("STARTING VIDEO PROCESSING")
            self.console_output.append(f"Video: {os.path.basename(video_path)}")
            self.console_output.append(f"Frame range: {options['start_frame']} to {options['end_frame']} ({total_frames} frames)")
            self.console_output.append(f"Time offset: {options['time_offset']} seconds")
            self.console_output.append(f"Estimated processing time: ~{estimated_time:.1f} seconds (plus rendering)")
            self.console_output.append(f"Output directory: {output_dir}")
            self.console_output.append("=" * 50)
            
            # Update UI
            self.export_status.setText("Processing video frames...")
            self.export_status.setStyleSheet("color: blue")
            
            # Force UI update
            QApplication.processEvents()
            
            # Process video
            result = processor.process_video(
                output_dir, 
                time_offset=options['time_offset'],
                start_frame=options['start_frame'],
                end_frame=options['end_frame']
            )
            
            if not result:
                self.export_status.setText("Error: Failed to process video frames")
                self.export_status.setStyleSheet("color: red")
                self.console_output.append("Error: Failed to process video frames. Check console for details.")
                return
            
            # Update UI
            self.export_status.setText("Creating MP4 video from frames...")
            self.export_status.setStyleSheet("color: blue")
            
            # Force UI update
            QApplication.processEvents()
            
            # Create output video
            self.console_output.append("-" * 50)
            self.console_output.append("Frames processed successfully. Creating output video...")
            output_video = processor.create_output_video()
            
            # Show success message with clickable link
            success_message = f"Video processing completed successfully!"
            self.export_status.setText(success_message)
            self.export_status.setStyleSheet("color: green")
            
            # Add detailed completion message to console
            self.console_output.append("=" * 50)
            self.console_output.append("VIDEO PROCESSING COMPLETED SUCCESSFULLY")
            self.console_output.append(f"Output video: {output_video}")
            self.console_output.append(f"File size: {os.path.getsize(output_video) / (1024*1024):.2f} MB")
            self.console_output.append("=" * 50)
            
            # Ask if user wants to open the video
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage(f"Video saved to: {output_video}", 10000)
            
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            self.export_status.setText(error_msg)
            self.export_status.setStyleSheet("color: red")
            
            # Add detailed error information to console
            self.console_output.append("=" * 50)
            self.console_output.append("ERROR DURING VIDEO PROCESSING")
            self.console_output.append(error_msg)
            self.console_output.append("Stack trace:")
            self.console_output.append(traceback.format_exc())
    
    # Add method to class
    gui_class.generate_overlay_frames_clicked = generate_overlay_frames_clicked
    return gui_class


# Testing function for standalone usage
def main():
    """Test function for video processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process flight video with telemetry overlay")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--data", required=True, help="Path to flight data CSV file")
    parser.add_argument("--output", help="Output directory for processed frames")
    parser.add_argument("--offset", type=float, default=0, help="Time offset in seconds")
    
    args = parser.parse_args()
    
    # Load flight data
    flight_data = pd.read_csv(args.data, sep=';')
    
    # Create video processor
    processor = VideoProcessor(
        video_path=args.video,
        flight_data=flight_data
    )
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(project_dir, "processed_frames")
    
    # Process video
    processor.process_video(output_dir, time_offset=args.offset)
    
    # Create output video and clean frames afterward
    processor.create_output_video(clean_frames=True)


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


class VideoProcessor:
    """Class for processing video and overlaying flight telemetry data"""
    
    def __init__(self, video_path=None, flight_data=None):
        """Initialize with video path and flight data DataFrame"""
        self.video_path = video_path
        self.flight_data = flight_data
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.duration = 0
        self.frame_width = 0
        self.frame_height = 0
        self.output_dir = None
        
        # Font configuration
        try:
            font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'arial.ttf')
            if os.path.exists(font_path):
                self.font_path = font_path
            else:
                # Try to use system fonts as fallback
                from matplotlib import rcParams
                self.font_path = rcParams['font.family']
                print(f"Font not found at {font_path}, using system font: {self.font_path}")
        except Exception as e:
            print(f"Error finding fonts: {e}. Falling back to default font.")
            self.font_path = None
            
        # Default color scheme (green for HUD elements)
        self.hud_color = (0, 255, 0)  # Green in BGR
        self.warning_color = (0, 0, 255)  # Red in BGR
        
    def load_video(self):
        """Load video file and extract properties"""
        if not self.video_path or not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {self.video_path}")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Loaded video: {os.path.basename(self.video_path)}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps:.2f}")
        print(f"Duration: {self.duration:.2f} seconds ({self.frame_count} frames)")
        
        return True
        
    def sync_video_with_data(self, time_offset=0):
        """
        Synchronize video with flight data using a time offset.
        
        Args:
            time_offset: Offset in seconds to align video with telemetry data
                         (positive if video starts after telemetry, negative if before)
        """
        if self.flight_data is None:
            raise ValueError("Flight data not loaded. Load flight data first.")
            
        # Create a mapping from video frame number to flight data index
        # Using video time (in seconds) + offset to find the corresponding data points
        self.frame_to_data_map = {}
        
        # Get the start time in seconds from the flight data
        min_time = self.flight_data['second'].min()
        
        for frame_num in range(self.frame_count):
            # Calculate the time in seconds for this frame
            frame_time = frame_num / self.fps
            
            # Adjust with offset and find the closest time in flight data
            adjusted_time = frame_time + time_offset + min_time
            
            # Find the closest time in the flight data
            closest_idx = (self.flight_data['second'] - adjusted_time).abs().idxmin()
            self.frame_to_data_map[frame_num] = closest_idx
            
        print(f"Synchronized video with flight data (offset: {time_offset} seconds)")
        return True
        
    def get_data_for_frame(self, frame_num):
        """Get flight data corresponding to a specific video frame"""
        if frame_num not in self.frame_to_data_map:
            return None
            
        data_idx = self.frame_to_data_map[frame_num]
        return self.flight_data.loc[data_idx]
        
    def create_hud_overlay(self, frame, frame_data):
        """
        Create a HUD (Heads-Up Display) overlay with flight data
        
        Args:
            frame: Video frame as numpy array
            frame_data: Pandas Series with flight data for this frame
        
        Returns:
            Frame with HUD overlay
        """
        # Convert OpenCV BGR frame to PIL Image for easier text handling
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Try to use custom font if available
        try:
            # Use different font sizes for different elements
            large_font_size = max(16, int(self.frame_height / 30))
            medium_font_size = max(14, int(self.frame_height / 40))
            small_font_size = max(12, int(self.frame_height / 50))
            
            if self.font_path:
                large_font = ImageFont.truetype(self.font_path, large_font_size)
                medium_font = ImageFont.truetype(self.font_path, medium_font_size)
                small_font = ImageFont.truetype(self.font_path, small_font_size)
            else:
                # Fall back to default font
                large_font = ImageFont.load_default()
                medium_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        except Exception as e:
            print(f"Error loading fonts: {e}")
            large_font = ImageFont.load_default()
            medium_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Convert HUD color from BGR to RGB for PIL
        hud_color_rgb = (self.hud_color[2], self.hud_color[1], self.hud_color[0])
        warning_color_rgb = (self.warning_color[2], self.warning_color[1], self.warning_color[0])
        
        # Extract the required data
        speed = frame_data.get('speed[m/s]', 0)
        altitude = frame_data.get('Approxaltitude[m]', 0)
        vertical_speed = frame_data.get('vertical_speed', 0)
        
        # Display speed directly in m/s without conversion to km/h
        
        # Format speed and altitude with proper units
        speed_text = f"SPD: {speed:.1f} m/s"
        altitude_text = f"ALT: {altitude:.1f} m"
        vspeed_text = f"V/S: {vertical_speed:.1f} m/s"
        
        # Define positions for the HUD elements
        # Speed on the left side
        speed_pos = (20, 30)
        # Altitude on the right side
        altitude_pos = (self.frame_width - 150, 30)
        # Vertical speed below altitude
        vspeed_pos = (self.frame_width - 150, 70)
        
        # Draw the HUD elements
        draw.text(speed_pos, speed_text, fill=hud_color_rgb, font=large_font)
        draw.text(altitude_pos, altitude_text, fill=hud_color_rgb, font=large_font)
        draw.text(vspeed_pos, vspeed_text, fill=hud_color_rgb, font=medium_font)
        
        # Add time counter at the bottom
        time_seconds = frame_data.get('second', 0)
        # Properly convert numpy.int64 to Python int using item()
        if hasattr(time_seconds, 'item'):
            time_seconds = time_seconds.item()
        time_formatted = str(timedelta(seconds=time_seconds))
        if "." in time_formatted:
            time_formatted = time_formatted.split(".")[0]  # Remove microseconds
        time_pos = (20, self.frame_height - 40)
        draw.text(time_pos, f"TIME: {time_formatted}", fill=hud_color_rgb, font=medium_font)
        
        # Draw artificial horizon line (simplified)
        horizon_length = int(self.frame_width * 0.4)
        horizon_center_x = self.frame_width // 2
        horizon_center_y = self.frame_height // 2
        
        # Draw central crosshair/reticle
        crosshair_size = 20
        line_thickness = 2
        
        # Horizontal line
        draw.line(
            (horizon_center_x - crosshair_size, horizon_center_y, 
             horizon_center_x + crosshair_size, horizon_center_y),
            fill=hud_color_rgb, width=line_thickness
        )
        
        # Vertical line
        draw.line(
            (horizon_center_x, horizon_center_y - crosshair_size, 
             horizon_center_x, horizon_center_y + crosshair_size),
            fill=hud_color_rgb, width=line_thickness
        )
        
        # Draw altitude indicator bars on right side (like a ladder)
        ladder_height = 200
        ladder_right = self.frame_width - 20
        ladder_center_y = self.frame_height // 2
        
        # Draw the ladder ticks
        for i in range(-2, 3):
            tick_y = ladder_center_y - i * (ladder_height // 5)
            tick_alt = altitude + i * 10  # 10m increments
            
            # Draw tick line
            tick_length = 15 if i == 0 else 10
            draw.line(
                (ladder_right - tick_length, tick_y, ladder_right, tick_y),
                fill=hud_color_rgb, width=line_thickness
            )
            
            # Only label every other tick to avoid clutter
            if i % 1 == 0:
                # Draw altitude label
                tick_label = f"{tick_alt:.0f}"
                draw.text(
                    (ladder_right - tick_length - 40, tick_y - 10),
                    tick_label, fill=hud_color_rgb, font=small_font
                )
        
        # Convert back to OpenCV format
        result_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return result_frame
    
    def process_video(self, output_dir, time_offset=0, start_frame=0, end_frame=None):
        """
        Process video with flight data overlay and save frames
        
        Args:
            output_dir: Directory to save output frames
            time_offset: Time offset in seconds to sync video with flight data
            start_frame: First frame to process (default: 0)
            end_frame: Last frame to process (default: all frames)
        """
        # Create a progress logger for this operation
        logger = ProgressLogger(prefix="[Frame Processing]")
        logger.log("Starting video processing...")
        
        try:
            if not self.cap:
                logger.log("Loading video file...")
                if not self.load_video():
                    logger.log("Failed to load video file", show_elapsed=True)
                    return False
                    
            if self.flight_data is None:
                raise ValueError("Flight data not loaded. Load flight data first.")
                
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
            
            # Synchronize video with flight data
            logger.log(f"Synchronizing video with flight data (offset: {time_offset}s)...")
            self.sync_video_with_data(time_offset)
            
            # Set frame range
            if end_frame is None or end_frame > self.frame_count:
                end_frame = self.frame_count
                
            # Validate frame range
            if start_frame >= end_frame:
                raise ValueError(f"Invalid frame range: start ({start_frame}) must be less than end ({end_frame})")
                
            # Set video position to start frame
            if start_frame > 0:
                logger.log(f"Setting start position to frame {start_frame}...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
            total_frames = end_frame - start_frame
            logger.log(f"Processing {total_frames} frames from {start_frame} to {end_frame-1}...")
            
            # Process frames
            processed_count = 0
            error_count = 0
            max_errors = min(50, total_frames // 10)  # Allow up to 10% errors or 50, whichever is smaller
            skipped_frames = []
            
            for frame_num in range(start_frame, end_frame):
                try:
                    # Read frame
                    ret, frame = self.cap.read()
                    if not ret:
                        error_count += 1
                        skipped_frames.append(frame_num)
                        if error_count <= 5:  # Only show first few errors to avoid spam
                            logger.log(f"Failed to read frame {frame_num}")
                        if error_count > max_errors:
                            logger.log(f"Too many frame reading errors ({error_count}). Stopping processing.")
                            break
                        continue
                        
                    # Get flight data for this frame
                    frame_data = self.get_data_for_frame(frame_num)
                    if frame_data is None:
                        if processed_count < 5:  # Only show warning for first few frames
                            logger.log(f"Warning: No flight data for frame {frame_num}, using empty data")
                        # Create empty data if none exists to avoid errors
                        frame_data = pd.Series({
                            'second': 0,
                            'speed[m/s]': 0,
                            'Approxaltitude[m]': 0,
                            'vertical_speed': 0
                        })
                        
                    # Create overlay
                    processed_frame = self.create_hud_overlay(frame, frame_data)
                    
                    # Save frame
                    output_file = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
                    cv2.imwrite(output_file, processed_frame)
                    
                    # Update progress
                    processed_count += 1
                    logger.progress(processed_count, total_frames)
                    
                except Exception as e:
                    error_count += 1
                    skipped_frames.append(frame_num)
                    if error_count <= 5:  # Only show first few errors to avoid spam
                        logger.log(f"Error processing frame {frame_num}: {str(e)}")
                    if error_count > max_errors:
                        logger.log(f"Too many processing errors ({error_count}). Stopping processing.")
                        break
            
            # Final progress update
            success_rate = (processed_count / total_frames) * 100 if total_frames > 0 else 0
            logger.log(f"Processed {processed_count} of {total_frames} frames ({success_rate:.1f}%). "
                      f"Output saved to {output_dir}", show_elapsed=True)
            
            if error_count > 0:
                logger.log(f"Warning: Encountered {error_count} errors during processing.")
                if len(skipped_frames) > 0:
                    skip_ranges = []
                    start = skipped_frames[0]
                    prev = start
                    
                    for i in range(1, len(skipped_frames)):
                        if skipped_frames[i] != prev + 1:
                            if start == prev:
                                skip_ranges.append(f"{start}")
                            else:
                                skip_ranges.append(f"{start}-{prev}")
                            start = skipped_frames[i]
                        prev = skipped_frames[i]
                    
                    if start == prev:
                        skip_ranges.append(f"{start}")
                    else:
                        skip_ranges.append(f"{start}-{prev}")
                        
                    skip_str = ", ".join(skip_ranges[:5])
                    if len(skip_ranges) > 5:
                        skip_str += f", ... and {len(skip_ranges) - 5} more ranges"
                    
                    logger.log(f"Skipped frames: {skip_str}")
                
            return processed_count > 0  # Return True if at least some frames were processed
            
        except Exception as e:
            logger.log(f"Error during video processing: {str(e)}")
            logger.log(traceback.format_exc())
            raise
        
    def create_output_video(self, output_path=None, frame_rate=None):
        """
        Create video from processed frames
        
        Args:
            output_path: Path to save output video (default: same as input with _overlay suffix)
            frame_rate: Frame rate of output video (default: same as input)
        """
        logger = ProgressLogger(prefix="[Video Creation]")
        logger.log("Starting output video creation...")
        video_writer = None
        
        try:
            if self.output_dir is None or not os.path.exists(self.output_dir):
                raise ValueError("No processed frames found. Process video first.")
                
            # Set default output path if not provided
            if output_path is None:
                if self.video_path:
                    filename = os.path.splitext(os.path.basename(self.video_path))[0]
                    output_path = os.path.join(os.path.dirname(self.video_path), f"{filename}_overlay.mp4")
                else:
                    output_path = os.path.join(self.output_dir, "output_overlay.mp4")
                    
            # Ensure we're not trying to overwrite the original video
            if self.video_path and os.path.abspath(output_path) == os.path.abspath(self.video_path):
                logger.log("Warning: Attempting to overwrite original video. Changing output path.")
                base_dir = os.path.dirname(self.video_path)
                filename = os.path.splitext(os.path.basename(self.video_path))[0]
                output_path = os.path.join(base_dir, f"{filename}_overlay_new.mp4")
                
            # Set default frame rate if not provided
            if frame_rate is None:
                frame_rate = self.fps
                
            # Verify frame rate
            if frame_rate <= 0:
                logger.log(f"Invalid frame rate detected: {frame_rate}, using 30 fps as fallback")
                frame_rate = 30.0
                
            # Get list of frame files
            frame_files = sorted([f for f in os.listdir(self.output_dir) if f.startswith("frame_") and f.endswith(".jpg")])
            if not frame_files:
                raise ValueError("No processed frames found in output directory.")
                
            logger.log(f"Found {len(frame_files)} frames to compile into video")
                
            # Read first frame to get dimensions
            first_frame_path = os.path.join(self.output_dir, frame_files[0])
            first_frame = cv2.imread(first_frame_path)
            if first_frame is None:
                raise ValueError(f"Failed to read first frame: {first_frame_path}")
                
            height, width, _ = first_frame.shape
            logger.log(f"Frame dimensions: {width}x{height}, frame rate: {frame_rate:.2f} fps")
            
            # List of codecs to try in order of preference
            codec_options = [
                ('avc1', 'H.264'),  # H.264, widely supported
                ('mp4v', 'MPEG-4'),  # MPEG-4, good fallback
                ('XVID', 'XVID'),    # XVID, good compatibility
                ('MJPG', 'Motion JPEG')  # MJPG, widely supported but larger files
            ]
            
            # Try codecs until one works
            video_writer = None
            used_codec_name = "unknown"
            
            for codec, codec_name in codec_options:
                try:
                    # Delete existing file if it exists to avoid issues
                    if os.path.exists(output_path):
                        os.remove(output_path)
                        
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    temp_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
                    
                    if temp_writer.isOpened():
                        video_writer = temp_writer
                        used_codec_name = codec_name
                        logger.log(f"Using {codec_name} codec ({codec})")
                        break
                    else:
                        temp_writer.release()
                except Exception as e:
                    logger.log(f"Codec {codec_name} ({codec}) failed: {str(e)}")
            
            if video_writer is None or not video_writer.isOpened():
                # Last desperate attempt with default codec
                logger.log("All codecs failed, trying one last attempt with default codec")
                fourcc = 0
                video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
                
            if not video_writer.isOpened():
                error_msg = (
                    "Failed to create video writer with any codec. "
                    "This might be due to:\n"
                    "1. Missing codecs in your OpenCV installation\n"
                    "2. Permission issues writing to the output directory\n"
                    "3. Insufficient disk space\n"
                    "Try installing OpenCV with full codec support or use a different output format."
                )
                raise ValueError(error_msg)
            
            # Add frames to video with progress reporting
            total_frames = len(frame_files)
            error_count = 0
            frame_count = 0
            
            # Process in chunks for better memory management
            chunk_size = 100  # Process 100 frames at a time
            
            for start_idx in range(0, total_frames, chunk_size):
                end_idx = min(start_idx + chunk_size, total_frames)
                chunk_frames = frame_files[start_idx:end_idx]
                
                for frame_file in chunk_frames:
                    try:
                        frame_path = os.path.join(self.output_dir, frame_file)
                        frame = cv2.imread(frame_path)
                        
                        if frame is not None:
                            video_writer.write(frame)
                            frame_count += 1
                        else:
                            error_count += 1
                            if error_count <= 5:  # Only log first few errors
                                logger.log(f"Warning: Could not read frame {frame_file}")
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:
                            logger.log(f"Error processing frame {frame_file}: {str(e)}")
                
                # Update progress after each chunk
                logger.progress(min(end_idx, total_frames), total_frames, 
                               f"Writing frames using {used_codec_name} codec")
            
            # Verify the output file was created and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 1000:  # At least 1KB
                    logger.log(f"Video creation completed. File size: {file_size/1024/1024:.2f} MB", show_elapsed=True)
                    logger.log(f"Output video saved to: {output_path}")
                    
                    if error_count > 0:
                        logger.log(f"Warning: {error_count} frames could not be processed during video creation")
                        
                    return output_path
                else:
                    raise ValueError(f"Output video file is too small ({file_size} bytes). Video creation likely failed.")
            else:
                raise ValueError("Output video file was not created.")
                
        except Exception as e:
            logger.log(f"Error creating output video: {str(e)}")
            logger.log(traceback.format_exc())

            # Provide troubleshooting information
            logger.log("\nTroubleshooting tips:")
            logger.log("1. Ensure your OpenCV installation includes video codec support")
            logger.log("2. Try installing ffmpeg if not already installed")
            logger.log("3. Make sure you have write permissions to the output directory")
            logger.log("4. Try a different output format or location")
            logger.log("5. If on Windows, try installing the K-Lite Codec Pack")

            raise
        finally:
            # Properly close the video writer if it exists
            if video_writer is not None and hasattr(video_writer, 'release'):
                try:
                    video_writer.release()
                    logger.log("Video writer resources released")
                except Exception as e:
                    logger.log(f"Warning: Error while releasing video writer: {e}")


def add_gui_integration(gui_class):
    """
    Add video processing functionality to GUI class
    
    Args:
        gui_class: GUI class to add functionality to (must be a class, not an instance)
    """

    # Only keep one version of the method with the updated implementation
    def generate_overlay_frames_clicked(self):
        """Method to handle 'Generate Telemetry Overlay' button click"""
        if not hasattr(self, 'performance_calculator') or self.performance_calculator is None:
            self.console_output.append("Please load flight data first")
            return
            
        # Ask user for video file
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if not video_path:
            return
            
        # Update UI to show processing status
        self.export_status.setText("Preparing video processing...")
        self.export_status.setStyleSheet("color: blue")
        
        # Make sure console tab is visible
        if hasattr(self, 'main_tabs'):
            for i in range(self.main_tabs.count()):
                if self.main_tabs.tabText(i) == "Console":
                    self.main_tabs.setCurrentIndex(i)
                    break
        
        try:
            self.console_output.append(f"Selected video: {os.path.basename(video_path)}")
            
            # Create video processor
            processor = VideoProcessor(
                video_path=video_path,
                flight_data=self.performance_calculator.df
            )
            
            # Update status
            self.export_status.setText("Loading video file...")
            self.export_status.setStyleSheet("color: blue")
            
            # Load video to get properties
            processor.load_video()
            
            # Show options dialog
            options_dialog = VideoOptionsDialog(
                parent=self,
                video_duration=processor.duration,
                frame_count=processor.frame_count
            )
            
            if not options_dialog.exec():
                self.console_output.append("Video processing cancelled by user")
                self.export_status.setText("Video processing cancelled")
                self.export_status.setStyleSheet("color: gray")
                return
                
            # Get selected options
            options = options_dialog.get_options()
            
            # Use the project directory for frames
            project_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(project_dir, "processed_frames")
            
            # Calculate expected duration and display
            total_frames = options['end_frame'] - options['start_frame']
            estimated_time = total_frames / processor.fps  # seconds per frame
            
            # Process video with selected options
            self.console_output.append("=" * 50)
            self.console_output.append("STARTING VIDEO PROCESSING")
            self.console_output.append(f"Video: {os.path.basename(video_path)}")
            self.console_output.append(f"Frame range: {options['start_frame']} to {options['end_frame']} ({total_frames} frames)")
            self.console_output.append(f"Time offset: {options['time_offset']} seconds")
            self.console_output.append(f"Estimated processing time: ~{estimated_time:.1f} seconds (plus rendering)")
            self.console_output.append(f"Output directory: {output_dir}")
            self.console_output.append("=" * 50)
            
            # Update UI
            self.export_status.setText("Processing video frames...")
            self.export_status.setStyleSheet("color: blue")
            
            # Force UI update
            QApplication.processEvents()
            
            # Process video
            result = processor.process_video(
                output_dir, 
                time_offset=options['time_offset'],
                start_frame=options['start_frame'],
                end_frame=options['end_frame']
            )
            
            if not result:
                self.export_status.setText("Error: Failed to process video frames")
                self.export_status.setStyleSheet("color: red")
                self.console_output.append("Error: Failed to process video frames. Check console for details.")
                return
            
            # Update UI
            self.export_status.setText("Creating MP4 video from frames...")
            self.export_status.setStyleSheet("color: blue")
            
            # Force UI update
            QApplication.processEvents()
            
            # Create output video
            self.console_output.append("-" * 50)
            self.console_output.append("Frames processed successfully. Creating output video...")
            output_video = processor.create_output_video()
            
            # Show success message with clickable link
            success_message = f"Video processing completed successfully!"
            self.export_status.setText(success_message)
            self.export_status.setStyleSheet("color: green")
            
            # Add detailed completion message to console
            self.console_output.append("=" * 50)
            self.console_output.append("VIDEO PROCESSING COMPLETED SUCCESSFULLY")
            self.console_output.append(f"Output video: {output_video}")
            self.console_output.append(f"File size: {os.path.getsize(output_video) / (1024*1024):.2f} MB")
            self.console_output.append("=" * 50)
            
            # Ask if user wants to open the video
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage(f"Video saved to: {output_video}", 10000)
            
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            self.export_status.setText(error_msg)
            self.export_status.setStyleSheet("color: red")
            
            # Add detailed error information to console
            self.console_output.append("=" * 50)
            self.console_output.append("ERROR DURING VIDEO PROCESSING")
            self.console_output.append(error_msg)
            self.console_output.append("Stack trace:")
            self.console_output.append(traceback.format_exc())
            self.console_output.append("=" * 50)
            
            # Troubleshooting tips
            self.console_output.append("\nTroubleshooting tips:")
            self.console_output.append("1. Make sure your video file is not corrupted")
            self.console_output.append("2. Try processing a smaller section of the video")
            self.console_output.append("3. Check that you have sufficient disk space")
            self.console_output.append("4. Ensure OpenCV is installed with video codec support")


    # Add method to class
    gui_class.generate_overlay_frames_clicked = generate_overlay_frames_clicked
    return gui_class


# Testing function for standalone usage
def main():
    """Test function for video processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process flight video with telemetry overlay")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--data", required=True, help="Path to flight data CSV file")
    parser.add_argument("--output", help="Output directory for processed frames")
    parser.add_argument("--offset", type=float, default=0, help="Time offset in seconds")
    
    args = parser.parse_args()
    
    # Load flight data
    flight_data = pd.read_csv(args.data, sep=';')
    
    # Create video processor
    processor = VideoProcessor(
        video_path=args.video,
        flight_data=flight_data
    )
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(os.path.dirname(args.video), "processed_frames")
    
    # Process video
    processor.process_video(output_dir, time_offset=args.offset)
    
    # Create output video
    processor.create_output_video()


if __name__ == "__main__":
    main()