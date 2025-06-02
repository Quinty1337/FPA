from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                            QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGroupBox,
                            QLineEdit, QGridLayout, QDoubleSpinBox, QComboBox, 
                            QTextEdit, QTabWidget, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
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
        self.setMinimumSize(1200, 800)  # Increased size for better tab display

        # Create central widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Create file selection section at top (remains outside tabs)
        file_group = QGroupBox("Flight Data")
        file_layout = QHBoxLayout()

        self.file_label = QLabel("No file selected")
        self.select_button = QPushButton("Select CSV File")
        self.select_button.clicked.connect(self.select_file)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_button)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # Create main tab widget
        self.main_tabs = QTabWidget()
        main_layout.addWidget(self.main_tabs)

        # Initialize data storage
        self.current_file = None
        self.processed_data = None
        self.performance_calculator = None
        
        # Set up tabs
        self.setup_aircraft_params_tab()
        self.setup_performance_metrics_tab()
        self.setup_export_tab()
        self.setup_graphs_tab()
        
        # Create console tab at bottom for logging
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
        
        # Set up console redirection
        self.stdout_redirector = ConsoleRedirector(self.console_output)
        sys.stdout = self.stdout_redirector
        
        # Add console as a tab
        self.main_tabs.addTab(console_tab, "Console")
        
        # Set initial parameter values
        self.set_initial_parameters()

    def setup_aircraft_params_tab(self):
        """Set up the Aircraft Parameters tab"""
        # Create tab
        aircraft_tab = QWidget()
        aircraft_layout = QVBoxLayout(aircraft_tab)
        
        # Create scrollable area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Constants section
        constants_group = QGroupBox("Calculation Constants")
        constants_layout = QGridLayout()
        
        # Add constant parameters (air density, gravity)
        constants_layout.addWidget(QLabel("Air Density (kg/m³):"), 0, 0)
        self.air_density = QDoubleSpinBox()
        self.air_density.setRange(0.0, 2.0)
        self.air_density.setValue(1.225)
        self.air_density.setSingleStep(0.001)
        self.air_density.setDecimals(3)
        constants_layout.addWidget(self.air_density, 0, 1)
        
        constants_layout.addWidget(QLabel("Gravity (m/s²):"), 0, 2)
        self.gravity = QDoubleSpinBox()
        self.gravity.setRange(9.0, 10.0)
        self.gravity.setValue(9.81)
        self.gravity.setSingleStep(0.01)
        self.gravity.setDecimals(2)
        constants_layout.addWidget(self.gravity, 0, 3)
        
        constants_group.setLayout(constants_layout)
        scroll_layout.addWidget(constants_group)
        
        # Aircraft parameters section
        aircraft_group = QGroupBox("Aircraft Parameters")
        aircraft_layout_grid = QGridLayout()

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
            aircraft_layout_grid.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            aircraft_layout_grid.addWidget(widget, i // 2, (i % 2) * 2 + 1)

        aircraft_group.setLayout(aircraft_layout_grid)
        scroll_layout.addWidget(aircraft_group)
        
        # Create update button
        self.update_params_button = QPushButton("Update Parameters")
        self.update_params_button.clicked.connect(self.update_aircraft_parameters)
        self.update_params_button.setMinimumHeight(40)  # Make button more prominent
        scroll_layout.addWidget(self.update_params_button)
        
        # Status label for parameter updates
        self.params_status_label = QLabel("")
        scroll_layout.addWidget(self.params_status_label)
        
        # Add stretch to push everything to the top
        scroll_layout.addStretch()
        
        # Set the scroll content and add to layout
        scroll_area.setWidget(scroll_content)
        aircraft_layout.addWidget(scroll_area)
        
        # Add tab to main tabs
        self.main_tabs.addTab(aircraft_tab, "Aircraft Parameters")

    def setup_performance_metrics_tab(self):
        """Set up the Performance Metrics tab"""
        # Create tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        # Create scrollable area for metrics
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Basic Flight Statistics
        stats_group = QGroupBox("Basic Flight Statistics")
        stats_layout = QVBoxLayout()
        stats_grid = QHBoxLayout()

        # Basic stats labels
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
        scroll_layout.addWidget(stats_group)

        # Add performance metrics section
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

        # Add the grids to the layout
        perf_layout.addLayout(perf_grid1)
        perf_layout.addLayout(perf_grid2)
        perf_layout.addLayout(perf_grid3)
        perf_layout.addLayout(perf_grid4)
        perf_group.setLayout(perf_layout)
        scroll_layout.addWidget(perf_group)
        
        # Add stretch to push everything to the top
        scroll_layout.addStretch()
        
        # Set the scroll content and add to layout
        scroll_area.setWidget(scroll_content)
        metrics_layout.addWidget(scroll_area)
        
        # Add tab to main tabs
        self.main_tabs.addTab(metrics_tab, "Performance Metrics")

    def setup_export_tab(self):
        """Set up the Export tab for KML and telemetry video"""
        # Create tab
        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)
        
        # KML Export Section
        kml_group = QGroupBox("KML Export")
        kml_layout = QVBoxLayout()
        
        kml_description = QLabel("Export flight data to KML format for viewing in Google Earth or other mapping applications.")
        kml_description.setWordWrap(True)
        kml_layout.addWidget(kml_description)
        
        kml_button_layout = QHBoxLayout()
        self.export_button = QPushButton("Export to KML")
        self.export_button.clicked.connect(self.export_kml)
        self.export_button.setEnabled(False)
        self.export_button.setMinimumHeight(40)
        
        self.kml_status = QLabel("")
        
        kml_button_layout.addWidget(self.export_button)
        kml_button_layout.addWidget(self.kml_status)
        kml_button_layout.addStretch()
        
        kml_layout.addLayout(kml_button_layout)
        kml_group.setLayout(kml_layout)
        export_layout.addWidget(kml_group)
        
        # Telemetry Video Section
        video_group = QGroupBox("Telemetry Video Overlay")
        video_layout = QVBoxLayout()
        
        video_description = QLabel("Generate video frames with telemetry data overlay for creating videos of your flight.")
        video_description.setWordWrap(True)
        video_layout.addWidget(video_description)
        
        video_button_layout = QHBoxLayout()
        self.overlay_button = QPushButton("Generate Telemetry Overlay")
        self.overlay_button.clicked.connect(self.generate_overlay_frames_clicked)
        self.overlay_button.setEnabled(False)
        self.overlay_button.setMinimumHeight(40)
        
        self.export_status = QLabel("")
        
        video_button_layout.addWidget(self.overlay_button)
        video_button_layout.addWidget(self.export_status)
        video_button_layout.addStretch()
        
        video_layout.addLayout(video_button_layout)
        video_group.setLayout(video_layout)
        export_layout.addWidget(video_group)
        
        # Add stretch to push everything to the top
        export_layout.addStretch()
        
        # Add tab to main tabs
        self.main_tabs.addTab(export_tab, "Export")

    def setup_graphs_tab(self):
        """Set up the Graphs tab"""
        # Create tab
        graphs_tab = QWidget()
        graphs_layout = QVBoxLayout(graphs_tab)
        
        # Add graph selector
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
        graphs_layout.addLayout(graph_selector_layout)
        
        # Create figure with subplots
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        graphs_layout.addWidget(self.canvas)
        
        # Add tab to main tabs
        self.main_tabs.addTab(graphs_tab, "Graphs")




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
            
                # Update calculator with calculation constants if available
                if hasattr(self, 'air_density') and hasattr(self, 'gravity'):
                    self.performance_calculator.rho = self.air_density.value()
                    self.performance_calculator.g = self.gravity.value()
                
                # Update calculator parameters
                self.performance_calculator.update_aircraft_parameters(params)
            
                # Recalculate and update display
                self.update_performance_stats()
                
                # Show success message
                self.params_status_label.setText("Parameters updated successfully!")
                self.params_status_label.setStyleSheet("color: green")
                
                # Switch to performance metrics tab to show updated results
                self.main_tabs.setCurrentIndex(1)  # Index 1 is the performance metrics tab
            
        except Exception as e:
            # Show error message
            self.params_status_label.setText(f"Error updating parameters: {str(e)}")
            self.params_status_label.setStyleSheet("color: red")

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
            self.overlay_button.setEnabled(True)
            
            # Switch to the Aircraft Parameters tab after loading file
            self.main_tabs.setCurrentIndex(0)  # Index 0 is the aircraft parameters tab
    
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
            
            # Display success message
            self.console_output.append(f"Successfully loaded and processed: {os.path.basename(file_path)}")
    
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            self.console_output.append(error_msg)
            self.export_status.setText(error_msg)
            self.export_status.setStyleSheet("color: red")
            
            # Switch to console tab to show error
            self.main_tabs.setCurrentIndex(4)  # Index 4 is the console tab
    
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
            
            # Check if Altitude[m] column exists, otherwise try Approxaltitude[m]
            if 'Altitude[m]' in self.processed_data.columns:
                altitude_column = 'Altitude[m]'
            else:
                altitude_column = 'Approxaltitude[m]'
                
            self.max_altitude_label.setText(
                f"Max Altitude: {self.processed_data[altitude_column].max():.1f} m")
            self.max_speed_label.setText(
                f"Max Speed: {self.processed_data['speed[m/s]'].max():.1f} m/s")
            self.flight_duration_label.setText(
                f"Flight Duration: {flight_duration:.1f} s")
            self.avg_altitude_label.setText(
                f"Avg Altitude: {self.processed_data[altitude_column].mean():.1f} m")
    
    def export_kml(self):
        if self.current_file:
            try:
                output_file = os.path.join(os.path.dirname(self.current_file), 'output.kml')
                convert_to_kml(self.current_file, output_file=output_file)
                
                # Update both status labels
                success_message = f"KML exported successfully to: {output_file}"
                self.kml_status.setText("KML exported successfully!")
                self.kml_status.setStyleSheet("color: green")
                self.console_output.append(success_message)
                
                # Switch to console tab to show the output path
                self.main_tabs.setCurrentIndex(4)  # Index 4 is the console tab
                
            except Exception as e:
                error_message = f"Export failed: {str(e)}"
                self.kml_status.setText(error_message)
                self.kml_status.setStyleSheet("color: red")
                self.console_output.append(error_message)

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

            # Switch to the Graphs tab when a graph is selected
            self.main_tabs.setCurrentIndex(3)  # Index 3 is the graphs tab
    
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
        
    def closeEvent(self, event):
        """Restore original stdout when closing the application"""
        sys.stdout = self.stdout_redirector.old_stdout
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = FlightAnalyzerGUI()
    window.show()
    
    # Display welcome message in console
    window.console_output.append("Welcome to Flight Performance Analyzer")
    window.console_output.append("Please select a CSV file to begin analysis")
    
    # Make console tab active on startup
    window.main_tabs.setCurrentIndex(4)  # Index 4 is the console tab
    
    sys.exit(app.exec())

# Apply the GUI integration from the Videoprocessor module
try:
    videoprocessor.add_gui_integration(FlightAnalyzerGUI)
    print("Video processing functionality integrated successfully")
except Exception as e:
    print(f"Failed to integrate video processing: {e}")
    # Fallback: Create minimal method if integration fails
    def generate_overlay_frames_clicked(self):
        self.console_output.append("Video processing module not available")
    FlightAnalyzerGUI.generate_overlay_frames_clicked = generate_overlay_frames_clicked

if __name__ == "__main__":
    main()