from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                            QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGroupBox,
                            QLineEdit, QGridLayout, QDoubleSpinBox, QComboBox, 
                            QTextEdit, QTabWidget)  # Add QTabWidget and QTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont  # Add this import
import sys
import os
import pandas as pd
from kmlconverter import convert_to_kml, clean_coordinate
from performancecalc import FlightPerformanceCalculator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import videoprocessor

from io import StringIO
import contextlib


class ConsoleRedirector(StringIO):
    def __init__(self, text_widget):
        super(ConsoleRedirector, self).__init__()
        self.text_widget = text_widget
        self.old_stdout = sys.stdout

    def write(self, text):
        # Write to the original stdout
        self.old_stdout.write(text)
        # Also write to the text widget
        self.text_widget.append(text)

    def flush(self):
        self.old_stdout.flush()

class FlightAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flight Performance Analyzer")
        self.setMinimumSize(1000, 800)  # Increased size to accommodate new metrics

        # Create central widget and main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create file selection section
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout()

        self.file_label = QLabel("No file selected")
        self.select_button = QPushButton("Select CSV File")
        self.select_button.clicked.connect(self.select_file)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_button)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        aircraft_group = QGroupBox("Aircraft Parameters")
        aircraft_layout = QGridLayout()

        # Weight parameters
        self.empty_weight = self._create_parameter_input("Empty Weight (kg)", 0.0, 1000.0, 1.0)
        self.max_weight = self._create_parameter_input("Max Weight (kg)", 0.0, 1000.0, 1.0)
        
        # Geometric parameters
        self.wingspan = self._create_parameter_input("Wingspan (m)", 0.0, 10.0, 0.1)
        self.wing_area = self._create_parameter_input("Wing Area (m²)", 0.0, 10.0, 0.01)
        self.aspect_ratio = self._create_parameter_input("Aspect Ratio", 0.0, 20.0, 0.1)
        
        # Aerodynamic parameters
        self.cd0 = self._create_parameter_input("CD0", 0.0, 1.0, 0.001)
        self.cl_max = self._create_parameter_input("CL max", 0.0, 2.0, 0.1)
        self.oswald_efficiency = self._create_parameter_input("Oswald Efficiency", 0.0, 1.0, 0.01)

        # Create update button
        self.update_params_button = QPushButton("Update Parameters")
        self.update_params_button.clicked.connect(self.update_aircraft_parameters)

        # Add all parameters to layout
        params = [
            ("Empty Weight (kg):", self.empty_weight),
            ("Max Weight (kg):", self.max_weight),
            ("Wingspan (m):", self.wingspan),
            ("Wing Area (m²):", self.wing_area),
            ("Aspect Ratio:", self.aspect_ratio),
            ("CD0:", self.cd0),
            ("CL max:", self.cl_max),
            ("Oswald Efficiency:", self.oswald_efficiency)
        ]

        for i, (label, widget) in enumerate(params):
            aircraft_layout.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            aircraft_layout.addWidget(widget, i // 2, (i % 2) * 2 + 1)

        # Add update button at the bottom
        aircraft_layout.addWidget(self.update_params_button, (len(params) // 2) + 1, 0, 1, 4)

        aircraft_group.setLayout(aircraft_layout)
        layout.addWidget(aircraft_group)

        # Update stats section
        stats_group = QGroupBox("Basic Flight Statistics")
        stats_layout = QVBoxLayout()
        stats_grid = QHBoxLayout()

        # Basic stats labels remain the same
        self.max_altitude_label = QLabel("Max Altitude: N/A")
        self.max_speed_label = QLabel("Max Speed: N/A")
        self.flight_duration_label = QLabel("Flight Duration: N/A")
        self.avg_altitude_label = QLabel("Average Altitude: N/A")

        stats_grid.addWidget(self.max_altitude_label)
        stats_grid.addWidget(self.max_speed_label)
        stats_grid.addWidget(self.flight_duration_label)
        stats_grid.addWidget(self.avg_altitude_label)

        stats_layout.addLayout(stats_grid)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Add new performance metrics section
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QVBoxLayout()

        # Create grid layouts for performance metrics
        perf_grid1 = QHBoxLayout()
        perf_grid2 = QHBoxLayout()
        perf_grid3 = QHBoxLayout()

        # Performance metric labels
        self.min_descent_rate_label = QLabel("Min Rate of Descent: N/A")
        self.ld_max_label = QLabel("L/D Max: N/A")
        self.max_climb_rate_label = QLabel("Max Rate of Climb: N/A")
        self.climb_angle_label = QLabel("Max Climb Angle: N/A")
        self.descent_angle_label = QLabel("Descent Angle: N/A")
        self.stall_speed_label = QLabel("Stall Speed: N/A")
        self.takeoff_length_label = QLabel("Predicted Takeoff Length: N/A")
        self.landing_length_label = QLabel("Predicted Landing Length: N/A")
        # Add new labels for actual values
        self.actual_takeoff_length_label = QLabel("Actual Takeoff Length: N/A")
        self.actual_landing_length_label = QLabel("Actual Landing Length: N/A")
        self.load_factor_label = QLabel("Max Load Factor: N/A")

        # Add labels to grid layouts
        perf_grid1.addWidget(self.min_descent_rate_label)
        perf_grid1.addWidget(self.max_climb_rate_label)
        perf_grid1.addWidget(self.ld_max_label)

        perf_grid2.addWidget(self.climb_angle_label)
        perf_grid2.addWidget(self.descent_angle_label)
        perf_grid2.addWidget(self.load_factor_label)

        perf_grid3.addWidget(self.stall_speed_label)
        perf_grid3.addWidget(self.takeoff_length_label)
        perf_grid3.addWidget(self.landing_length_label)

        # Add a new grid for the actual values
        perf_grid4 = QHBoxLayout()
        perf_grid4.addWidget(self.actual_takeoff_length_label)
        perf_grid4.addWidget(self.actual_landing_length_label)

        # Add the new grid to the layout
        perf_layout.addLayout(perf_grid1)
        perf_layout.addLayout(perf_grid2)
        perf_layout.addLayout(perf_grid3)
        perf_layout.addLayout(perf_grid4)
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        # Create export section
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout()

        self.export_button = QPushButton("Export to KML")
        self.export_button.clicked.connect(self.export_kml)
        self.export_button.setEnabled(False)

        self.overlay_button = QPushButton("Generate Telemetry Overlay")
        self.overlay_button.clicked.connect(self.generate_overlay_frames_clicked)
        self.overlay_button.setEnabled(False)

        self.export_status = QLabel("")

        export_layout.addWidget(self.export_button)
        export_layout.addWidget(self.overlay_button)
        export_layout.addWidget(self.export_status)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Create tab widget for graphs and console
        tab_widget = QTabWidget()

        # First tab for plots
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)

        # Add graph selector and plots as before
        graph_selector_layout = QHBoxLayout()
        self.graph_selector = QComboBox()
        self.graph_selector.addItems([
            "All Graphs", 
            "Altitude Profile", 
            "Speed Profile", 
            "Vertical Speed", 
            "Original vs Processed Altitude"
        ])
        self.graph_selector.currentIndexChanged.connect(self.update_graph_display)
        graph_selector_layout.addWidget(QLabel("Select Graph:"))
        graph_selector_layout.addWidget(self.graph_selector)
        graph_selector_layout.addStretch()
        plot_layout.addLayout(graph_selector_layout)

        # Create figure with subplots
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        # Second tab for console
        console_tab = QWidget()
        console_layout = QVBoxLayout(console_tab)
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFont(QFont("Courier New", 9))
        self.console_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        # Clear button
        clear_console_button = QPushButton("Clear Console")
        clear_console_button.clicked.connect(self.clear_console)
        console_layout.addWidget(self.console_output)
        console_layout.addWidget(clear_console_button)

        # Add tabs
        tab_widget.addTab(plot_tab, "Graphs")
        tab_widget.addTab(console_tab, "Console")

        # Add tab widget to layout
        layout.addWidget(tab_widget)

        # Set up console redirection
        self.stdout_redirector = ConsoleRedirector(self.console_output)
        sys.stdout = self.stdout_redirector

        # Data storage
        self.current_file = None
        self.processed_data = None
        self.performance_calculator = None
    
        # Set initial parameter values
        self.set_initial_parameters()

    def _create_parameter_input(self, name, min_val, max_val, step):
        """Helper method to create parameter input fields"""
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(3)
        return spinbox

    def update_aircraft_parameters(self):
        """Update aircraft parameters and recalculate performance"""
        try:
            if self.performance_calculator is not None:
                # Validate parameters before updating
                if self.wing_area.value() <= 0:
                    raise ValueError("Wing area must be greater than 0")
                if self.wingspan.value() <= 0:
                    raise ValueError("Wingspan must be greater than 0")
                if self.empty_weight.value() <= 0:
                    raise ValueError("Empty weight must be greater than 0")
                if self.max_weight.value() <= self.empty_weight.value():
                    raise ValueError("Max weight must be greater than empty weight")
            
                params = {
                    'empty_weight': self.empty_weight.value(),
                    'max_weight': self.max_weight.value(),
                    'wingspan': self.wingspan.value(),
                    'wing_area': self.wing_area.value(),
                    'aspect_ratio': self.aspect_ratio.value(),
                    'cd0': self.cd0.value(),
                    'cl_max': self.cl_max.value(),
                    'oswald_efficiency': self.oswald_efficiency.value()
                }
            
                # Update calculator parameters
                self.performance_calculator.update_aircraft_parameters(params)
            
                # Recalculate and update display
                self.update_performance_stats()
            
        except Exception as e:
            # Create a status label if it doesn't exist
            if not hasattr(self, 'status_label'):
                self.status_label = QLabel()
                self.layout().addWidget(self.status_label)
            
            # Show error message
            self.status_label.setText(f"Error updating parameters: {str(e)}")
            self.status_label.setStyleSheet("color: red")

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_name:
            self.current_file = file_name
            self.file_label.setText(os.path.basename(file_name))
            self.process_file(file_name)
            self.export_button.setEnabled(True)

    def create_plots(self):
        """Create visualization plots with takeoff and landing points"""
        if self.processed_data is None:
            return

        self.figure.clear()
        
        # Get the selected graph index
        selected_graph = self.graph_selector.currentIndex()
        
        if selected_graph == 0:
            # "All Graphs" - Show the 2x2 grid
            ax1 = self.figure.add_subplot(221)  # Altitude vs Time
            ax2 = self.figure.add_subplot(222)  # Speed vs Time
            ax3 = self.figure.add_subplot(223)  # Vertical Speed vs Time
            ax4 = self.figure.add_subplot(224)  # Original vs Processed Altitude
            
            self.plot_altitude_profile(ax1)
            self.plot_speed_profile(ax2)
            self.plot_vertical_speed(ax3)
            self.plot_original_vs_processed(ax4)
        
        elif selected_graph == 1:
            # "Altitude Profile" - Single full-width graph
            ax = self.figure.add_subplot(111)
            self.plot_altitude_profile(ax)
        
        elif selected_graph == 2:
            # "Speed Profile" - Single full-width graph
            ax = self.figure.add_subplot(111)
            self.plot_speed_profile(ax)
        
        elif selected_graph == 3:
            # "Vertical Speed" - Single full-width graph
            ax = self.figure.add_subplot(111)
            self.plot_vertical_speed(ax)
        
        elif selected_graph == 4:
            # "Original vs Processed Altitude" - Single full-width graph
            ax = self.figure.add_subplot(111)
            self.plot_original_vs_processed(ax)

        self.figure.tight_layout()
        self.canvas.draw()

    def process_file(self, file_path):
        try:
            # Create performance calculator instance
            self.performance_calculator = FlightPerformanceCalculator(file_path)
            self.processed_data = self.performance_calculator.df

            # Calculate runway distances to get the indices
            self.performance_calculator.calculate_actual_takeoff_length()
            self.performance_calculator.calculate_actual_landing_length()

            # Update both basic and performance statistics
            self.update_stats()
            self.update_performance_stats()
            self.create_plots()  # This will now display the points

        except Exception as e:
            self.export_status.setText(f"Error processing file: {str(e)}")
            self.export_status.setStyleSheet("color: red")

    def update_performance_stats(self):
        """Update the performance metrics display"""
        if self.performance_calculator is not None:
            perf_data = self.performance_calculator.get_performance_summary()
            
            self.min_descent_rate_label.setText(
                f"Min Rate of Descent: {perf_data['min_descent_rate']:.2f} m/s")
            self.ld_max_label.setText(
                f"L/D Max: {perf_data['ld_max']:.2f}")
            self.max_climb_rate_label.setText(
                f"Max Rate of Climb: {perf_data['max_climb_rate']:.2f} m/s")
            self.climb_angle_label.setText(
                f"Max Climb Angle: {perf_data['climb_angle_max']:.2f}°")
            self.descent_angle_label.setText(
                f"Descent Angle: {perf_data['descent_angle']:.2f}°")
        
        if perf_data['stall_speed']:
            self.stall_speed_label.setText(
                f"Stall Speed: {perf_data['stall_speed']:.2f} m/s")
        
        # Update with predicted runway lengths
        self.takeoff_length_label.setText(
            f"Predicted Takeoff Length: {perf_data['predicted_takeoff_runway_length']:.2f} m")
        self.landing_length_label.setText(
            f"Predicted Landing Length: {perf_data['predicted_landing_runway_length']:.2f} m")
        
        # Add display for actual runway lengths
        self.actual_takeoff_length_label.setText(
            f"Actual Takeoff Length: {perf_data['actual_takeoff_runway_length']:.2f} m")
        self.actual_landing_length_label.setText(
            f"Actual Landing Length: {perf_data['actual_landing_runway_length']:.2f} m")
        
        if perf_data['load_factor_max']:
            self.load_factor_label.setText(
                f"Max Load Factor: {perf_data['load_factor_max']:.2f} g")

    def update_stats(self):
        if self.processed_data is not None:
            flight_duration = self.performance_calculator.get_flight_duration()
            
            self.max_altitude_label.setText(
                f"Max Altitude: {self.processed_data['Altitude[m]'].max():.1f} m")
            self.max_speed_label.setText(
                f"Max Speed: {self.processed_data['speed[m/s]'].max():.1f} m/s")
            self.flight_duration_label.setText(
                f"Flight Duration: {flight_duration:.1f} s")
            self.avg_altitude_label.setText(
                f"Avg Altitude: {self.processed_data['Altitude[m]'].mean():.1f} m")

    def export_kml(self):
        if self.current_file:
            try:
                output_file = os.path.join(os.path.dirname(self.current_file), 'output.kml')
                convert_to_kml(self.current_file, output_file=output_file)
                self.export_status.setText("KML exported successfully!")
                self.export_status.setStyleSheet("color: green")
            except Exception as e:
                self.export_status.setText(f"Export failed: {str(e)}")
                self.export_status.setStyleSheet("color: red")

    def set_initial_parameters(self):
        """Set initial default values for aircraft parameters"""
        self.empty_weight.setValue(24.0)  # Example default value
        self.max_weight.setValue(24.999)    # Example default value
        self.wingspan.setValue(3.5)        # Example default value
        self.wing_area.setValue(1.45)       # Example default value
        self.aspect_ratio.setValue(8.448)    # Example default value
        self.cd0.setValue(0.175)           # Example default value
        self.cl_max.setValue(1.205)          # Example default value
        self.oswald_efficiency.setValue(0.85)  # Example default value

    def update_graph_display(self):
        """Update the displayed graph based on the selector"""
        if self.processed_data is not None:
            self.create_plots()

    def create_plots(self):
        """Create visualization plots with takeoff and landing points"""
        if self.processed_data is None:
            return

        self.figure.clear()
        
        # Get the selected graph index
        selected_graph = self.graph_selector.currentIndex()
        
        if selected_graph == 0:
            # "All Graphs" - Show the 2x2 grid
            ax1 = self.figure.add_subplot(221)  # Altitude vs Time
            ax2 = self.figure.add_subplot(222)  # Speed vs Time
            ax3 = self.figure.add_subplot(223)  # Vertical Speed vs Time
            ax4 = self.figure.add_subplot(224)  # Original vs Processed Altitude
            
            self.plot_altitude_profile(ax1)
            self.plot_speed_profile(ax2)
            self.plot_vertical_speed(ax3)
            self.plot_original_vs_processed(ax4)
        
        elif selected_graph == 1:
            # "Altitude Profile" - Single full-width graph
            ax = self.figure.add_subplot(111)
            self.plot_altitude_profile(ax)
        
        elif selected_graph == 2:
            # "Speed Profile" - Single full-width graph
            ax = self.figure.add_subplot(111)
            self.plot_speed_profile(ax)
        
        elif selected_graph == 3:
            # "Vertical Speed" - Single full-width graph
            ax = self.figure.add_subplot(111)
            self.plot_vertical_speed(ax)
        
        elif selected_graph == 4:
            # "Original vs Processed Altitude" - Single full-width graph
            ax = self.figure.add_subplot(111)
            self.plot_original_vs_processed(ax)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_altitude_profile(self, ax):
        """Plot the altitude profile on the given axis"""
        ax.plot(self.processed_data['second'], self.processed_data['Approxaltitude[m]'], 'b-')
        ax.set_title('Processed Altitude Profile with Runway Points')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m)')
        ax.grid(True)
        
        # Add takeoff and landing points if available
        if hasattr(self.performance_calculator, 'takeoff_start_idx') and self.performance_calculator.takeoff_start_idx is not None:
            start_time = self.processed_data.loc[self.performance_calculator.takeoff_start_idx, 'second']
            start_alt = self.processed_data.loc[self.performance_calculator.takeoff_start_idx, 'Approxaltitude[m]']
            ax.scatter(start_time, start_alt, color='blue', s=80, marker='o', 
                   label='Takeoff Start', zorder=5)
                   
        if hasattr(self.performance_calculator, 'takeoff_end_idx') and self.performance_calculator.takeoff_end_idx is not None:
            end_time = self.processed_data.loc[self.performance_calculator.takeoff_end_idx, 'second']
            end_alt = self.processed_data.loc[self.performance_calculator.takeoff_end_idx, 'Approxaltitude[m]']
            ax.scatter(end_time, end_alt, color='red', s=80, marker='o', 
                   label='Takeoff End', zorder=5)
                   
        if hasattr(self.performance_calculator, 'landing_start_idx') and self.performance_calculator.landing_start_idx is not None:
            start_time = self.processed_data.loc[self.performance_calculator.landing_start_idx, 'second']
            start_alt = self.processed_data.loc[self.performance_calculator.landing_start_idx, 'Approxaltitude[m]']
            ax.scatter(start_time, start_alt, color='green', s=80, marker='o', 
                   label='Landing Start', zorder=5)
                   
        if hasattr(self.performance_calculator, 'landing_end_idx') and self.performance_calculator.landing_end_idx is not None:
            end_time = self.processed_data.loc[self.performance_calculator.landing_end_idx, 'second']
            end_alt = self.processed_data.loc[self.performance_calculator.landing_end_idx, 'Approxaltitude[m]']
            ax.scatter(end_time, end_alt, color='yellow', s=80, marker='o', 
                   label='Landing End', zorder=5)

        # Add legend for the runway points
        ax.legend(loc='upper right')

    def plot_speed_profile(self, ax):
        """Plot the speed profile on the given axis"""
        ax.plot(self.processed_data['second'], self.processed_data['speed[m/s]'], 'r-')
        ax.set_title('Speed Profile')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.grid(True)

    def plot_vertical_speed(self, ax):
        """Plot the vertical speed on the given axis"""
        ax.plot(self.processed_data['second'], self.processed_data['vertical_speed'], 'g-')
        ax.set_title('Vertical Speed Profile')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Vertical Speed (m/s)')
        ax.grid(True)

    def plot_original_vs_processed(self, ax):
        """Plot the original vs processed altitude"""
        # Check if 'original_altitude' exists in the data
        if 'original_altitude' in self.processed_data.columns:
            ax.plot(self.processed_data['second'], self.processed_data['original_altitude'], 'r--', 
                alpha=0.5, label='Original')
            ax.plot(self.processed_data['second'], self.processed_data['Approxaltitude[m]'], 'b-', 
                label='Processed')
            ax.set_title('Original vs Processed Altitude')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Altitude (m)')
            ax.grid(True)
            ax.legend(loc='upper right')
        else:
            # If original_altitude is not available, just show processed
            self.plot_altitude_profile(ax)

    def clear_console(self):
        """Clear the console output"""
        self.console_output.clear()

class ConsoleRedirector(StringIO):
    def __init__(self, text_widget):
        super(ConsoleRedirector, self).__init__()
        self.text_widget = text_widget
        self.old_stdout = sys.stdout

    def write(self, text):
        # Write to the original stdout
        self.old_stdout.write(text)
        # Also write to the text widget
        self.text_widget.append(text)

    def flush(self):
        self.old_stdout.flush()

def clear_console(self):
    """Clear the console output text widget"""
    self.console_output.clear()

def closeEvent(self, event):
    """Restore original stdout when closing the application"""
    sys.stdout = self.stdout_redirector.old_stdout
    super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = FlightAnalyzerGUI()
    window.show()
    sys.exit(app.exec())

def process_data(self):
    """Process and prepare data for display"""
    if self.csv_file is None:
        return
        
    # Create performance calculator
    self.performance_calculator = FlightPerformanceCalculator(self.csv_file)
    
    # IMPORTANT CHANGE: Use the processed dataframe directly from the performance calculator
    # This ensures we're using the same corrected altitude data
    self.processed_data = self.performance_calculator.df
    
    # Update the display
    self.update_graph_display()
    self.update_performance_summary()

# Apply the GUI integration from the Videoprocessor module
try:
    videoprocessor.add_gui_integration(FlightAnalyzerGUI)
    print("Video processing functionality integrated successfully")
except Exception as e:
    print(f"Failed to integrate video processing: {e}")

if __name__ == "__main__":
    main()