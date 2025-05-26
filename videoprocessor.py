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

def add_gui_integration(parent_class):
    """Add overlay generation functionality to the GUI"""
    def generate_overlay_frames_clicked(self):
        # Check if flight data is loaded
        if self.performance_calculator is None:
            self.console_output.append("Error: Please load flight data first")
            return
        
        try:
            # Ask for output directory
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory for Overlay Frames",
                os.path.dirname(self.current_file) if self.current_file else ""
            )
            
            if not output_dir:
                return
                
            # Create output directory
            overlay_dir = os.path.join(output_dir, "telemetry_overlay")
            os.makedirs(overlay_dir, exist_ok=True)
            
            # Ask for frame rate and duration
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDialogButtonBox
            
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
            
            # Create processor and generate frames
            self.console_output.append("Generating overlay frames...")
            processor = VideoProcessor(self.current_file)
            processor.generate_overlay_frames(
                output_dir=overlay_dir,
                frame_rate=frame_rate,
                duration=duration
            )
            processor.create_instructions_file(overlay_dir)
            
            self.console_output.append(f"Overlay frames generated in {overlay_dir}")
            
        except Exception as e:
            self.console_output.append(f"Error generating overlay frames: {str(e)}")
    
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