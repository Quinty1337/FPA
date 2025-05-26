from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtWidgets import QFileDialog, QPushButton, QGroupBox
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import argparse
import subprocess
import tempfile
import shutil
import time

class VideoProcessor:
    def __init__(self, csv_path, output_dir=None):
        """
        Initialize the video processor with path to CSV data.
        
        Args:
            csv_path: Path to the CSV file with flight data
            output_dir: Directory for the output files (defaults to current directory)
        """
        self.csv_path = csv_path
        
        if output_dir is None:
            self.output_dir = os.path.dirname(csv_path)
        else:
            self.output_dir = output_dir
            
        # Load and process the data
        self._load_data()
        
        # Create fonts directory if needed
        self.fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
        os.makedirs(self.fonts_dir, exist_ok=True)
        
        # Check for font file or download a default one
        self.font_path = os.path.join(self.fonts_dir, 'arial.ttf')
        if not os.path.exists(self.font_path):
            # Use a system font if available
            system_fonts = [
                'C:/Windows/Fonts/arial.ttf',  # Windows
                '/Library/Fonts/Arial.ttf',    # Mac
                '/usr/share/fonts/truetype/freefont/FreeSans.ttf'  # Linux
            ]
            for font in system_fonts:
                if os.path.exists(font):
                    shutil.copy(font, self.font_path)
                    break
        
    def _load_data(self):
        """Load and process flight data from CSV"""
        try:
            # Load CSV file
            self.df = pd.read_csv(self.csv_path, sep=';')
            
            # Make sure time is in milliseconds
            if 'Time[ms]' in self.df.columns:
                self.df['time_ms'] = self.df['Time[ms]']
            else:
                raise ValueError("CSV file must contain 'Time[ms]' column")
                
            # Check for required columns
            required_columns = ['Approxaltitude[m]', 'speed[m/s]']
            for col in required_columns:
                if col not in self.df.columns:
                    raise ValueError(f"CSV file must contain '{col}' column")
                    
            # Convert timestamps to video time
            self.df['video_time'] = self.df['time_ms'] / 1000.0
            
            # Adjust starting time to match video
            min_time = self.df['video_time'].min()
            self.df['video_time'] = self.df['video_time'] - min_time
            
            print(f"Loaded {len(self.df)} data points from CSV")
            
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            sys.exit(1)
            
    def _get_data_at_time(self, video_time):
        """Get flight data for the current video time"""
        # Find the closest time in the dataframe
        closest_idx = (self.df['video_time'] - video_time).abs().idxmin()
        return self.df.loc[closest_idx]
        
    def create_overlay_image(self, video_time, width=1920, height=1080):
        """Create a transparent overlay image with telemetry data"""
        # Create a transparent image
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Get data for the current time
        data = self._get_data_at_time(video_time)
        
        # Load font - with fallback to default
        try:
            font_large = ImageFont.truetype(self.font_path, 48)
            font_small = ImageFont.truetype(self.font_path, 36)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Background rectangle for telemetry (top right)
        rect_width = 320
        rect_height = 160
        rect_x = width - rect_width - 10
        rect_y = 10
        
        # Draw semi-transparent black rectangle
        draw.rectangle(
            [(rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height)],
            fill=(0, 0, 0, 180)
        )
        
        # Get altitude and speed data
        altitude = data['Approxaltitude[m]']
        speed = data['speed[m/s]']
        
        # Convert speed to km/h for more intuitive display
        speed_kmh = speed * 3.6
        
        # Text position
        text_x = rect_x + 10
        
        # Draw altitude text
        draw.text(
            (text_x, rect_y + 20),
            f"ALT: {altitude:.1f} m",
            fill=(255, 255, 255, 255),
            font=font_large
        )
        
        # Draw speed text
        draw.text(
            (text_x, rect_y + 80),
            f"SPD: {speed_kmh:.1f} km/h",
            fill=(255, 255, 255, 255),
            font=font_large
        )
        
        # Add vertical speed if available
        if 'vertical_speed' in data:
            vert_speed = data['vertical_speed']
            draw.text(
                (text_x, rect_y + 130),
                f"Vertical: {vert_speed:.1f} m/s",
                fill=(255, 255, 255, 255),
                font=font_small
            )
        
        return overlay
        
    def generate_overlay_frames(self, output_dir=None, frame_rate=30, duration=None):
        """
        Generate PNG overlay frames that can be used with video editing software
        
        Args:
            output_dir: Directory to save overlay frames (default: temp directory)
            frame_rate: Frame rate of the target video (default: 30 fps)
            duration: Duration in seconds (default: use all data)
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'overlay_frames')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine duration
        if duration is None:
            duration = self.df['video_time'].max()
        
        # Calculate number of frames
        num_frames = int(duration * frame_rate)
        
        print(f"Generating {num_frames} overlay frames...")
        
        # Generate frames
        for i in range(num_frames):
            # Calculate current time
            video_time = i / frame_rate
            
            # Create overlay
            overlay = self.create_overlay_image(video_time)
            
            # Save frame
            frame_path = os.path.join(output_dir, f"overlay_{i:06d}.png")
            overlay.save(frame_path)
            
            # Show progress
            if i % 30 == 0:
                progress = (i / num_frames) * 100
                print(f"Progress: {progress:.1f}% ({i}/{num_frames})")
        
        print(f"Overlay frames generated in {output_dir}")
        return output_dir
        
    def create_instructions_file(self, overlay_dir):
        """Create a text file with instructions for using the overlay frames"""
        instructions_path = os.path.join(overlay_dir, "README.txt")
        
        with open(instructions_path, 'w') as f:
            f.write("Flight Telemetry Overlay Frames\n")
            f.write("================================\n\n")
            f.write("These PNG frames contain transparent overlays with flight data.\n")
            f.write("To use them in your video editor:\n\n")
            f.write("1. Import your GoPro video footage\n")
            f.write("2. Add these overlay frames as a sequence on a separate track above your video\n")
            f.write("3. Make sure the overlay track uses transparency/alpha blending\n\n")
            f.write("Common settings in popular video editors:\n")
            f.write("- **DaVinci Resolve**: Add images as a compound clip, set composite mode to 'Normal'\n")
            f.write("- **Adobe Premiere**: Import as image sequence, set blending mode to 'Normal'\n")
            f.write("- **Final Cut Pro**: Import as image sequence, add as connected clip\n\n")
            f.write("For best results, match the frame rate of your video when importing the sequence.\n")
        
        print(f"Instructions created at {instructions_path}")

    def create_standalone_video(self, frames_dir, output_video, frame_rate=30, width=1920, height=1080):
        """
        Create a standalone MP4 video from overlay frames with a black background
        
        Args:
            frames_dir: Directory containing overlay PNG frames
            output_video: Path to output MP4 file
            frame_rate: Frame rate of the output video
            width: Width of the output video
            height: Height of the output video
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import subprocess
            import os
            import shutil
            
            # Check if ffmpeg is available
            if shutil.which('ffmpeg') is None:
                print("Error: ffmpeg not found. Please install ffmpeg to create videos.")
                return False
            
            # Command to create video from PNG frames
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-r', str(frame_rate),  # Frame rate
                '-f', 'image2',  # Input format
                '-s', f'{width}x{height}',  # Size
                '-i', os.path.join(frames_dir, 'overlay_%06d.png'),  # Input pattern
                '-vcodec', 'libx264',  # Codec
                '-crf', '23',  # Quality (lower is better)
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                output_video
            ]
            
            # Run the command
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            return os.path.exists(output_video)
            
        except Exception as e:
            print(f"Error creating standalone video: {e}")
            return False

    def create_composite_video(self, frames_dir, input_video, output_video, frame_rate=30):
        """
        Create a composite MP4 video by overlaying frames on an input video
        
        Args:
            frames_dir: Directory containing overlay PNG frames
            input_video: Path to input video file
            output_video: Path to output MP4 file
            frame_rate: Frame rate of the output video
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import subprocess
            import os
            import shutil
            
            # Check if ffmpeg is available
            if shutil.which('ffmpeg') is None:
                print("Error: ffmpeg not found. Please install ffmpeg to create videos.")
                return False
            
            # Command to overlay PNG frames on input video
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', input_video,  # Input video
                '-r', str(frame_rate),  # Frame rate for overlay
                '-f', 'image2',  # Input format for overlay
                '-i', os.path.join(frames_dir, 'overlay_%06d.png'),  # Overlay pattern
                '-filter_complex', '[0:v][1:v]overlay=0:0',  # Overlay filter
                '-c:v', 'libx264',  # Video codec
                '-crf', '23',  # Quality (lower is better)
                '-c:a', 'copy',  # Copy audio
                output_video
            ]
            
            # Run the command
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            return os.path.exists(output_video)
        
        except Exception as e:
            print(f"Error creating composite video: {e}")
            return False

def add_gui_integration(parent_class):
    """Add overlay generation functionality to the GUI"""
    def generate_overlay_frames_clicked(self):
        # Check if flight data is loaded
        if self.performance_calculator is None:
            self.console_output.append("Error: Please load flight data first")
            return

        try:
            # Import required modules
            from PyQt6.QtWidgets import QFileDialog

            # Ask for output directory
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory for Overlay",
                os.path.dirname(self.current_file) if self.current_file else ""
            )
            
            if not output_dir:
                return
            
            # Create output directory
            overlay_dir = os.path.join(output_dir, "telemetry_overlay")
            os.makedirs(overlay_dir, exist_ok=True)
            
            # Ask for frame rate, duration, and output format
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDialogButtonBox, QRadioButton, QGroupBox, QFileDialog
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Overlay Settings")
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
            
            # Output format options
            format_group = QGroupBox("Output Format")
            format_layout = QVBoxLayout(format_group)
            
            png_radio = QRadioButton("PNG Sequence (for video editing)")
            mp4_radio = QRadioButton("MP4 Video")
            mp4_radio.setChecked(True)  # Default to MP4
            
            format_layout.addWidget(png_radio)
            format_layout.addWidget(mp4_radio)
            layout.addWidget(format_group)
            
            # Video input option (only shown when MP4 is selected)
            video_group = QGroupBox("Video Options")
            video_layout = QVBoxLayout(video_group)
            
            standalone_radio = QRadioButton("Standalone overlay (black background)")
            input_video_radio = QRadioButton("Composite with input video")
            input_video_radio.setChecked(True)  # Default to using input video
            
            video_path_layout = QHBoxLayout()
            video_path_label = QLabel("No video selected")
            video_path_button = QPushButton("Select Video")
            video_path = ""
            
            def select_video():
                nonlocal video_path
                path, _ = QFileDialog.getOpenFileName(
                    dialog,
                    "Select Input Video",
                    os.path.dirname(self.current_file) if self.current_file else "",
                    "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
                )
                if path:
                    video_path = path
                    video_path_label.setText(os.path.basename(path))
            
            video_path_button.clicked.connect(select_video)
            video_path_layout.addWidget(video_path_label)
            video_path_layout.addWidget(video_path_button)
            
            video_layout.addWidget(standalone_radio)
            video_layout.addWidget(input_video_radio)
            video_layout.addLayout(video_path_layout)
            
            layout.addWidget(video_group)
            
            # Show/hide video options based on selected format
            def update_video_options():
                video_group.setVisible(mp4_radio.isChecked())
                video_path_label.setVisible(input_video_radio.isChecked())
                video_path_button.setVisible(input_video_radio.isChecked())
            
            mp4_radio.toggled.connect(update_video_options)
            input_video_radio.toggled.connect(update_video_options)
            update_video_options()  # Initial state
            
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
            output_format = "mp4" if mp4_radio.isChecked() else "png"
            use_input_video = input_video_radio.isChecked() if mp4_radio.isChecked() else False
            
            # Create processor
            self.console_output.append("Generating overlay...")
            processor = VideoProcessor(self.current_file)
            
            if output_format == "png":
                # Generate PNG sequence
                processor.generate_overlay_frames(
                    output_dir=overlay_dir,
                    frame_rate=frame_rate,
                    duration=duration
                )
                processor.create_instructions_file(overlay_dir)
                self.console_output.append(f"Overlay frames generated in {overlay_dir}")
            
            else:  # MP4 output
                if use_input_video and not video_path:
                    self.console_output.append("Error: No input video selected")
                    return
                
                # Generate frames to a temporary directory
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    self.console_output.append("Generating overlay frames...")
                    frames_dir = processor.generate_overlay_frames(
                        output_dir=temp_dir,
                        frame_rate=frame_rate,
                        duration=duration
                    )
                    
                    # Create output MP4 path
                    output_mp4 = os.path.join(overlay_dir, "telemetry_overlay.mp4")
                    
                    # Compile to MP4
                    self.console_output.append("Creating MP4 video...")
                    
                    if use_input_video:
                        # Composite with input video
                        success = processor.create_composite_video(
                            frames_dir=frames_dir,
                            input_video=video_path,
                            output_video=output_mp4,
                            frame_rate=frame_rate
                        )
                    else:
                        # Standalone overlay video
                        success = processor.create_standalone_video(
                            frames_dir=frames_dir,
                            output_video=output_mp4,
                            frame_rate=frame_rate,
                            width=1920,
                            height=1080
                        )
                    
                    if success:
                        self.console_output.append(f"MP4 video created: {output_mp4}")
                    else:
                        self.console_output.append("Error: Failed to create MP4 video")
        
        except Exception as e:
            self.console_output.append(f"Error generating overlay: {str(e)}")

    # Add method to parent class
    setattr(parent_class, 'generate_overlay_frames_clicked', generate_overlay_frames_clicked)
    
    # Store original init method
    original_init = parent_class.__init__
    
    # Define new init to add the button
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        
        # Method 1: Try to find the Export group by its title
        found_group = None
        
        # Find all QGroupBoxes and check their titles
        all_group_boxes = self.findChildren(QGroupBox)
        for group_box in all_group_boxes:
            if group_box.title() == "Export":
                found_group = group_box
                break
        
        if found_group and found_group.layout():
            # Found the Export group and it has a layout
            export_layout = found_group.layout()
            
            # Add video processing button to export group
            self.overlay_button = QPushButton("Generate Telemetry Overlay")
            self.overlay_button.clicked.connect(self.generate_overlay_frames_clicked)
            export_layout.addWidget(self.overlay_button)
            self.overlay_button.setEnabled(False)
            
            # Update file selection method to enable overlay button
            original_select_file = self.select_file
            
            def new_select_file(*args, **kwargs):
                original_select_file(*args, **kwargs)
                self.overlay_button.setEnabled(True)
                
            self.select_file = new_select_file
            
            print("Telemetry overlay button added to Export group")
        else:
            # Method 2: Create a standalone button and add it after the Export group
            central_widget = self.centralWidget()
            if central_widget and central_widget.layout():
                main_layout = central_widget.layout()
                
                # Create a new button
                self.overlay_button = QPushButton("Generate Telemetry Overlay")
                self.overlay_button.clicked.connect(self.generate_overlay_frames_clicked)
                self.overlay_button.setEnabled(False)
                
                # Add the button directly to the main layout
                main_layout.addWidget(self.overlay_button)
                
                # Update file selection method to enable overlay button
                original_select_file = self.select_file
                
                def new_select_file(*args, **kwargs):
                    original_select_file(*args, **kwargs)
                    self.overlay_button.setEnabled(True)
                    
                self.select_file = new_select_file
                
                print("Telemetry overlay button added as standalone button")
            else:
                print("Warning: Could not find suitable parent for telemetry overlay button.")

def main():
    """Command-line interface for overlay generation"""
    parser = argparse.ArgumentParser(description="Generate telemetry overlay frames from flight data")
    parser.add_argument("csv", help="Path to CSV file with flight data")
    parser.add_argument("-o", "--output", help="Output directory for overlay frames")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frame rate (frames per second)")
    parser.add_argument("-d", "--duration", type=float, help="Duration in seconds (default: use all data)")
    
    args = parser.parse_args()
    
    processor = VideoProcessor(args.csv, args.output)
    overlay_dir = processor.generate_overlay_frames(
        output_dir=args.output,
        frame_rate=args.fps,
        duration=args.duration
    )
    processor.create_instructions_file(overlay_dir)

if __name__ == "__main__":
    if 'PyQt6.QtWidgets' in sys.modules:
        # If imported from GUI, add integration
        from PyQt6.QtWidgets import QGroupBox, QPushButton, QFileDialog, QVBoxLayout
        try:
            from gui import FlightAnalyzerGUI
            add_gui_integration(FlightAnalyzerGUI)
            print("Telemetry overlay integration added to GUI")
        except ImportError:
            print("GUI integration failed")
    else:
        # Run as standalone script
        main()
def generate_overlay_frames_clicked(self):
    # Check if flight data is loaded
    if self.performance_calculator is None:
        self.console_output.append("Error: Please load flight data first")
        return
    
    # Validate that we have a video file
    if not self.current_file:
        self.console_output.append("Error: No video file selected")
        return
        
    if not os.path.isfile(self.current_file):
        self.console_output.append(f"Error: File does not exist: {self.current_file}")
        return
    
    # Optional: Check file extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']  # Add relevant video formats
    if not any(self.current_file.lower().endswith(ext) for ext in valid_extensions):
        self.console_output.append(f"Warning: File may not be a supported video format: {self.current_file}")
        # Consider adding a confirmation dialog here
    
    try:
        # Remaining code...
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Overlay",
            os.path.dirname(self.current_file) if self.current_file else ""
        )
        
        if not output_dir:
            return
            
        # Create output directory
        overlay_dir = os.path.join(output_dir, "telemetry_overlay")
        os.makedirs(overlay_dir, exist_ok=True)
        
        # Ask for frame rate, duration, and output format
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDialogButtonBox, QRadioButton, QGroupBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Overlay Settings")
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
        
        # Output format options
        format_group = QGroupBox("Output Format")
        format_layout = QVBoxLayout(format_group)
        
        png_radio = QRadioButton("PNG Sequence (for video editing)")
        mp4_radio = QRadioButton("MP4 Video (using OpenCV)")
        mp4_radio.setChecked(True)  # Default to MP4
        
        format_layout.addWidget(png_radio)
        format_layout.addWidget(mp4_radio)
        layout.addWidget(format_group)
        
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
        output_format = "mp4" if mp4_radio.isChecked() else "png"
        
        # Create processor
        self.console_output.append("Generating overlay...")
        from videoprocessor import VideoProcessor
        processor = VideoProcessor(self.current_file)
        
        if output_format == "png":
            # Generate PNG sequence
            processor.generate_overlay_frames(
                output_dir=overlay_dir,
                frame_rate=frame_rate,
                duration=duration
            )
            processor.create_instructions_file(overlay_dir)
            self.console_output.append(f"Overlay frames generated in {overlay_dir}")
            
        else:  # MP4 output
            # Generate frames to a temporary directory
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                self.console_output.append("Generating overlay frames...")
                frames_dir = processor.generate_overlay_frames(
                    output_dir=temp_dir,
                    frame_rate=frame_rate,
                    duration=duration
                )
                
                # Create output MP4 path
                output_mp4 = os.path.join(overlay_dir, "telemetry_overlay.mp4")
                
                # Create MP4 using OpenCV
                self.console_output.append("Creating MP4 video using OpenCV...")
                
                # Use OpenCV instead of FFmpeg
                success = self.create_video_with_opencv(
                    frames_dir=frames_dir,
                    output_video=output_mp4,
                    frame_rate=frame_rate
                )
                
                if success:
                    self.console_output.append(f"MP4 video created: {output_mp4}")
                else:
                    self.console_output.append("Error: Failed to create MP4 video")
        
    except Exception as e:
        self.console_output.append(f"Error generating overlay: {str(e)}")
        import traceback
        self.console_output.append(traceback.format_exc())

def create_video_with_opencv(self, frames_dir, output_video, frame_rate=30):
    """
    Create an MP4 video from a sequence of overlay frames using OpenCV
    
    Args:
        frames_dir: Directory containing overlay PNG frames
        output_video: Path to output MP4 file
        frame_rate: Frame rate of the output video
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import cv2
        import os
        import glob
        
        # Get all PNG files and sort them numerically
        png_files = sorted(glob.glob(os.path.join(frames_dir, 'overlay_*.png')))
        
        if not png_files:
            self.console_output.append("No overlay frames found")
            return False
        
        # Read the first frame to get dimensions
        first_frame = cv2.imread(png_files[0])
        height, width, channels = first_frame.shape
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
        out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
        
        # Process each frame
        total_frames = len(png_files)
        self.console_output.append(f"Processing {total_frames} frames...")
        
        for i, file in enumerate(png_files):
            # Show progress every 10%
            if i % max(1, total_frames // 10) == 0:
                self.console_output.append(f"Processing frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
            
            # Read and write frame
            frame = cv2.imread(file)
            out.write(frame)
        
        # Release the VideoWriter
        out.release()
        
        # Verify the output file exists and has content
        if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
            self.console_output.append(f"Video saved to: {output_video}")
            return True
        else:
            self.console_output.append("Error: Output video file is empty or doesn't exist")
            return False
        
    except Exception as e:
        self.console_output.append(f"Error creating video with OpenCV: {str(e)}")
        import traceback
        self.console_output.append(traceback.format_exc())
        return False