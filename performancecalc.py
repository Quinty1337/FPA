import pandas as pd
import numpy as np
from kmlconverter import clean_coordinate

class FlightPerformanceCalculator:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, sep=';')

        # Default aircraft parameters
        self.aircraft_params = {
            'empty_weight': 0.0,
            'max_weight': 0.0,
            'wingspan': 0.0,
            'wing_area': 0.0,
            'aspect_ratio': 0.0,
            'cd0': 0.0,
            'cl_max': 0.0,
            'oswald_efficiency': 0.0
        }

        # Constants
        self.rho = 1.225  # Air density at sea level (kg/m³)
        self.g = 9.81  # Gravitational acceleration (m/s²)

        self.process_data()

    def process_data(self):
        """Process raw data and calculate additional parameters"""
        try:
            # Calculate time in seconds
            self.df['second'] = (self.df['Time[ms]'] / 1000).astype(int)

            # ALTITUDE CORRECTION
            # First, store original altitude
            self.df['original_altitude'] = self.df['Approxaltitude[m]'].copy()

            # Find start and end altitudes (average of first and last 10 points)
            start_alt = self.df['Approxaltitude[m]'].head(10).mean()
            end_alt = self.df['Approxaltitude[m]'].tail(10).mean()

            # Calculate the offset (average of start and end)
            offset = (start_alt + end_alt) / 2
            print(f"Altitude offset correction: {offset:.2f}m (start: {start_alt:.2f}m, end: {end_alt:.2f}m)")

            # Apply correction
            self.df['Approxaltitude[m]'] = self.df['Approxaltitude[m]'] - offset

            # Smooth altitude data with a simple rolling average (easier than median)
            try:
                self.df['Approxaltitude[m]'] = self.df['Approxaltitude[m]'].rolling(window=5, center=True).mean()
                # Fill missing values at beginning and end
                self.df['Approxaltitude[m]'] = self.df['Approxaltitude[m]'].fillna(self.df['original_altitude'] - offset)
            except Exception as e:
                print(f"Warning: Altitude smoothing failed: {e}")

            # Group by second for calculating vertical speed
            try:
                # Group by second to get one altitude value per second
                alt_by_second = self.df.groupby('second')['Approxaltitude[m]'].mean().reset_index()

                # Calculate vertical speed with simple difference (safer)
                alt_by_second['vertical_speed'] = alt_by_second['Approxaltitude[m]'].diff()

                # Apply simple smoothing to vertical speed
                alt_by_second['vertical_speed'] = alt_by_second['vertical_speed'].rolling(window=3, min_periods=1,
                                                                                          center=True).mean()

                # Apply realistic limits
                max_vs = 20.0  # max realistic vertical speed in m/s
                alt_by_second.loc[alt_by_second['vertical_speed'] > max_vs, 'vertical_speed'] = max_vs
                alt_by_second.loc[alt_by_second['vertical_speed'] < -max_vs, 'vertical_speed'] = -max_vs

                # Merge back to main dataframe
                self.df = pd.merge(self.df, alt_by_second[['second', 'vertical_speed']], on='second', how='left')
            except Exception as e:
                print(f"Warning: Vertical speed calculation failed: {e}")
                # Fallback: create empty vertical speed column
                self.df['vertical_speed'] = 0

            # Calculate acceleration magnitude
            try:
                self.df['acc_magnitude'] = np.sqrt(
                    self.df['accX[m/s2]'] ** 2 +
                    self.df['accY[m/s2]'] ** 2 +
                    self.df['accZ[m/s2]'] ** 2
                )

                # Calculate load factor
                self.df['load_factor'] = self.df['acc_magnitude'] / self.g
            except Exception as e:
                print(f"Warning: Acceleration calculation failed: {e}")
                self.df['acc_magnitude'] = 1.0 * self.g
                self.df['load_factor'] = 1.0

            # Clean up NaN values
            self.df = self.df.fillna(0)

            print("Data processing completed successfully")
        except Exception as e:
            print(f"Error in process_data: {e}")
            # Ensure basic columns exist to prevent further errors
            if 'vertical_speed' not in self.df.columns:
                self.df['vertical_speed'] = 0
            if 'load_factor' not in self.df.columns:
                self.df['load_factor'] = 1.0
            if 'acc_magnitude' not in self.df.columns:
                self.df['acc_magnitude'] = self.g

    def calculate_cl(self, velocity):
        """Calculate lift coefficient for given velocity"""
        if self.aircraft_params['wing_area'] and self.aircraft_params['max_weight']:
            weight_force = self.aircraft_params['max_weight'] * self.g
            dynamic_pressure = 0.5 * self.rho * velocity ** 2
            return weight_force / (dynamic_pressure * self.aircraft_params['wing_area'])
        return 0

    def calculate_cd(self, cl):
        """Calculate drag coefficient using drag polar"""
        if self.aircraft_params['oswald_efficiency'] and self.aircraft_params['aspect_ratio']:
            induced_drag = (cl ** 2) / (np.pi * self.aircraft_params['aspect_ratio'] *
                                        self.aircraft_params['oswald_efficiency'])
            return self.aircraft_params['cd0'] + induced_drag
        return 0

    def calculate_stall_speed(self):
        """Calculate stall speed"""
        if all(v > 0 for v in [self.aircraft_params['max_weight'],
                               self.aircraft_params['wing_area'],
                               self.aircraft_params['cl_max']]):
            return np.sqrt((2 * self.aircraft_params['max_weight'] * self.g) /
                           (self.rho * self.aircraft_params['wing_area'] *
                            self.aircraft_params['cl_max']))
        return 0

    def get_climb_angle(self):
        """Calculate maximum climb angle with error handling"""
        try:
            # Only consider points with meaningful speed and vertical speed
            valid_data = self.df[(self.df['speed[m/s]'] > 3) & (self.df['vertical_speed'] > 0.5)]

            if len(valid_data) > 0:
                # Calculate angles
                climb_angles = np.degrees(np.arctan2(valid_data['vertical_speed'], valid_data['speed[m/s]']))
                # Use 90th percentile for robustness
                return np.nanpercentile(climb_angles, 90)
            return 0
        except Exception as e:
            print(f"Error in get_climb_angle: {e}")
            return 0

    def get_descent_angle(self):
        """Calculate descent angle with error handling"""
        try:
            # Only consider points with meaningful speed and negative vertical speed
            valid_data = self.df[(self.df['speed[m/s]'] > 3) & (self.df['vertical_speed'] < -0.5)]

            if len(valid_data) > 0:
                # Calculate angles (use abs of vertical speed since we want positive angles)
                descent_angles = np.degrees(np.arctan2(abs(valid_data['vertical_speed']), valid_data['speed[m/s]']))
                # Use 90th percentile for robustness
                return np.nanpercentile(descent_angles, 90)
            return 0
        except Exception as e:
            print(f"Error in get_descent_angle: {e}")
            return 0

    def calculate_runway_length(self, phase):
        """Calculate runway length for takeoff or landing"""
        if phase == 'takeoff':
            # Simplified takeoff distance calculation
            if self.aircraft_params['cl_max'] and self.aircraft_params['max_weight']:
                v_lof = 1.2 * self.calculate_stall_speed()  # Liftoff speed
                acceleration = 0.5 * self.g  # Assumed acceleration
                return (v_lof ** 2) / (2 * acceleration)
        else:  # landing
            # Simplified landing distance calculation
            v_app = 1.3 * self.calculate_stall_speed()  # Approach speed
            deceleration = 0.4 * self.g  # Assumed deceleration
            return (v_app ** 2) / (2 * deceleration)
        return 0

    def get_flight_duration(self):
        """Calculate total flight duration in seconds"""
        return self.df['second'].max() - self.df['second'].min()

    def update_aircraft_parameters(self, params):
        """Update aircraft parameters with new values"""
        # Validate input parameters
        required_params = [
            'empty_weight', 'max_weight', 'wingspan', 'wing_area',
            'aspect_ratio', 'cd0', 'cl_max', 'oswald_efficiency'
        ]

        # Check if all required parameters are present
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")

        # Update parameters
        self.aircraft_params.update(params)

        # Validate updated parameters
        if self.aircraft_params['max_weight'] <= self.aircraft_params['empty_weight']:
            raise ValueError("Maximum weight must be greater than empty weight")
        if self.aircraft_params['wing_area'] <= 0:
            raise ValueError("Wing area must be positive")
        if self.aircraft_params['wingspan'] <= 0:
            raise ValueError("Wingspan must be positive")
        if self.aircraft_params['aspect_ratio'] <= 0:
            raise ValueError("Aspect ratio must be positive")
        if self.aircraft_params['oswald_efficiency'] <= 0 or self.aircraft_params['oswald_efficiency'] > 1:
            raise ValueError("Oswald efficiency must be between 0 and 1")
        if self.aircraft_params['cd0'] < 0:
            raise ValueError("CD0 cannot be negative")
        if self.aircraft_params['cl_max'] <= 0:
            raise ValueError("CL max must be positive")

    def calculate_actual_takeoff_length(self):
        """
        Calculate the actual takeoff length based on flight data.
        Takeoff starts when speed increases from 0 and ends when vertical acceleration becomes positive.
        """
        try:
            # Filter data for takeoff phase - assuming it's at the beginning of the flight
            takeoff_data = self.df.copy()

            # Find where speed starts increasing from near zero
            # Using a small threshold to account for sensor noise
            speed_threshold = 10.0  # m/s
            takeoff_start_idx = takeoff_data[takeoff_data['speed[m/s]'] > speed_threshold].index.min()

            if pd.isna(takeoff_start_idx):
                # Store indices for plotting
                self.takeoff_start_idx = None
                self.takeoff_end_idx = None
                return 0  # No valid takeoff data found

            # Find where vertical acceleration becomes positive (aircraft lifts off)
            # First calculate vertical acceleration if not already present
            if 'vertical_acc' not in takeoff_data.columns:
                # Calculate vertical acceleration as the derivative of vertical speed
                takeoff_data['vertical_acc'] = takeoff_data['vertical_speed'].diff() / (
                            takeoff_data['Time[ms]'].diff() / 1000)
                # Apply smoothing to reduce noise
                takeoff_data['vertical_acc'] = takeoff_data['vertical_acc'].rolling(window=5, center=True,
                                                                                    min_periods=1).mean()

            # Find where vertical acceleration becomes positive after takeoff start
            vert_acc_threshold = 1.0  # m/s²
            takeoff_data_after_start = takeoff_data.loc[takeoff_start_idx:]
            liftoff_idx = takeoff_data_after_start[
                takeoff_data_after_start['vertical_acc'] > vert_acc_threshold].index.min()

            if pd.isna(liftoff_idx):
                # Alternative: use where altitude starts increasing significantly
                alt_threshold = 10.0  # meters
                liftoff_idx = takeoff_data_after_start[
                    takeoff_data_after_start['Approxaltitude[m]'] > alt_threshold].index.min()

                if pd.isna(liftoff_idx):
                    # Store indices for plotting
                    self.takeoff_start_idx = takeoff_start_idx
                    self.takeoff_end_idx = None
                    return 0  # No valid liftoff point found

            # Store indices for plotting
            self.takeoff_start_idx = takeoff_start_idx
            self.takeoff_end_idx = liftoff_idx

            # Calculate the distance traveled during takeoff
            # Rest of the function remains the same...

            # If GPS coordinates are available, use them for more accurate distance
            if 'Longitude[deg]' in self.df.columns and 'Latitude[deg]' in self.df.columns:
                # Calculate distance using Haversine formula between points
                from geopy.distance import geodesic

                start_coords = (clean_coordinate(takeoff_data.loc[takeoff_start_idx, 'Latitude[deg]']),
                                clean_coordinate(takeoff_data.loc[takeoff_start_idx, 'Longitude[deg]']))
                end_coords = (clean_coordinate(takeoff_data.loc[liftoff_idx, 'Latitude[deg]']),
                              clean_coordinate(takeoff_data.loc[liftoff_idx, 'Longitude[deg]']))

                distance = geodesic(start_coords, end_coords).meters
                print("Takeoff Start Coords: ", start_coords,"Takeoff End Coords: ", end_coords)
                return distance
            else:
                # Estimate distance based on speed and time
                takeoff_duration = (takeoff_data.loc[liftoff_idx, 'Time[ms]'] -
                                    takeoff_data.loc[takeoff_start_idx, 'Time[ms]']) / 1000  # seconds

                # Use average speed during takeoff
                avg_speed = takeoff_data.loc[takeoff_start_idx:liftoff_idx, 'speed[m/s]'].mean()
                distance = avg_speed * takeoff_duration
                return distance

        except Exception as e:
            print(f"Error calculating actual takeoff length: {e}")
            # Store indices for plotting
            self.takeoff_start_idx = None
            self.takeoff_end_idx = None
            return 0

    def calculate_actual_landing_length(self):
        """
        Calculate the actual landing length based on flight data.
        Landing starts at altitude 0 (touchdown) and ends where speed is 0.
        """
        try:
            # Filter data for landing phase - assuming it's at the end of the flight
            landing_data = self.df.copy()
            takeoff_data = self.df.copy()

            # Find touchdown point (where altitude is close to zero)
            # Using a small threshold to account for sensor noise
            altitude_threshold = 1.0  # meters
            touchdown_candidates = landing_data[landing_data['Approxaltitude[m]'].abs() < altitude_threshold]

            if touchdown_candidates.empty:
                # Store indices for plotting
                self.landing_start_idx = None
                self.landing_end_idx = None
                return 0  # No valid touchdown data found

            # From potential touchdown points, find the one that's followed by continuous ground roll
            # This helps avoid false positives during approach
            touchdown_idx = None
            for idx in touchdown_candidates.index:
                if idx > len(landing_data) - 10 & idx < len(takeoff_data) + 10:  # Skip points near the end of the dataset
                    print(idx)
                    continue


                # Check if the next N points are also close to ground level
            next_points = landing_data.loc[idx:idx + 1, 'Approxaltitude[m]']
            if (next_points.abs() < altitude_threshold).all():
                touchdown_idx = idx


            if touchdown_idx is None:
                # If no suitable touchdown point found, use the last altitude crossing
                touchdown_idx = touchdown_candidates.index[-10] if len(touchdown_candidates) > 10 else \
                touchdown_candidates.index[0]
                print('alticross')

            # Find where speed approaches zero after touchdown
            speed_threshold = 2.0  # m/s
            landing_data_after_touchdown = landing_data.loc[touchdown_idx:]
            stop_candidates = landing_data_after_touchdown[landing_data_after_touchdown['speed[m/s]'] < speed_threshold]

            if stop_candidates.empty:
                # If no clear stopping point, use the last data point
                stop_idx = landing_data.index[-1]
            else:
                stop_idx = stop_candidates.index[0]

            # Store indices for plotting
            self.landing_start_idx = touchdown_idx
            self.landing_end_idx = stop_idx

            # Calculate the distance traveled during landing
            # Rest of the function remains the same...

            # If GPS coordinates are available, use them for more accurate distance
            if 'Longitude[deg]' in self.df.columns and 'Latitude[deg]' in self.df.columns:
                # Calculate distance using Haversine formula between points
                from geopy.distance import geodesic

                start_coords = (clean_coordinate(landing_data.loc[touchdown_idx, 'Latitude[deg]']),
                                clean_coordinate(landing_data.loc[touchdown_idx, 'Longitude[deg]']))
                end_coords = (clean_coordinate(landing_data.loc[stop_idx, 'Latitude[deg]']),
                              clean_coordinate(landing_data.loc[stop_idx, 'Longitude[deg]']))

                print("Landing Start Coords: ", start_coords, "Landing End Coords: ", end_coords)
                distance = geodesic(start_coords, end_coords).meters
                return distance
            else:
                # Estimate distance based on speed and time
                landing_duration = (landing_data.loc[stop_idx, 'Time[ms]'] -
                                    landing_data.loc[touchdown_idx, 'Time[ms]']) / 1000  # seconds

                # Use average speed during landing
                avg_speed = landing_data.loc[touchdown_idx:stop_idx, 'speed[m/s]'].mean()
                distance = avg_speed * landing_duration
                return distance
        except Exception as e:
            print(f"Error calculating actual landing length: {e}")
            # Store indices for plotting
            self.landing_start_idx = None
            self.landing_end_idx = None
            return 0

    def calculate_predicted_runway_length(self, phase):
        """Calculate predicted runway length for takeoff or landing based on theoretical formulas"""
        if phase == 'takeoff':
            # Simplified takeoff distance calculation
            if self.aircraft_params['cl_max'] and self.aircraft_params['max_weight']:
                v_lof = 1.2 * self.calculate_stall_speed()  # Liftoff speed
                acceleration = 0.5 * self.g  # Assumed acceleration
                return (v_lof ** 2) / (2 * acceleration)
        else:  # landing
            # Simplified landing distance calculation
            v_app = 1.3 * self.calculate_stall_speed()  # Approach speed
            deceleration = 0.4 * self.g  # Assumed deceleration
            return (v_app ** 2) / (2 * deceleration)
        return 0

    # Keep the original method for backwards compatibility
    def calculate_runway_length(self, phase):
        """Legacy method that calls the predicted runway length calculation"""
        return self.calculate_predicted_runway_length(phase)

    def get_performance_summary(self):
        """Get comprehensive performance summary with error handling"""
        try:
            # Basic statistics with fallbacks for errors
            try:
                avg_speed = self.df['speed[m/s]'].mean()
                max_speed = self.df['speed[m/s]'].max()
            except Exception:  # Added proper exception handler
                avg_speed = 0
                max_speed = 0

            # Calculate L/D ratio
            try:
                cruise_cl = self.calculate_cl(avg_speed)
                cruise_cd = self.calculate_cd(cruise_cl)
                ld_ratio = cruise_cl / cruise_cd if cruise_cd != 0 else 0
            except Exception:  # Added proper exception handler
                ld_ratio = 0

            # Get climb and descent rates - safely
            try:
                # Filter out zero speeds to avoid division issues
                positive_vs = self.df.loc[(self.df['vertical_speed'] > 0.5) &
                                          (self.df['speed[m/s]'] > 3), 'vertical_speed']
                negative_vs = self.df.loc[(self.df['vertical_speed'] < -0.5) &
                                          (self.df['speed[m/s]'] > 3), 'vertical_speed']

                # Use 90th percentile instead of max for robustness
                max_climb = positive_vs.quantile(0.9) if not positive_vs.empty else 0
                min_descent = abs(negative_vs.quantile(0.1)) if not negative_vs.empty else 0
            except Exception as e:
                print(f"Error calculating climb rates: {e}")
                max_climb = 0
                min_descent = 0

            # Get angles
            try:
                climb_angle = self.get_climb_angle()
                descent_angle = self.get_descent_angle()
            except:
                climb_angle = 0
                descent_angle = 0

            # Get stall speed
            try:
                stall_speed = self.calculate_stall_speed()
            except:
                stall_speed = 0

            # Get predicted runway lengths
            try:
                predicted_takeoff_length = self.calculate_predicted_runway_length('takeoff')
                predicted_landing_length = self.calculate_predicted_runway_length('landing')
            except:
                predicted_takeoff_length = 0
                predicted_landing_length = 0

            # Get actual runway lengths
            try:
                actual_takeoff_length = self.calculate_actual_takeoff_length()
                actual_landing_length = self.calculate_actual_landing_length()
            except Exception as e:
                print(f"Error calculating actual runway lengths: {e}")
                actual_takeoff_length = 0
                actual_landing_length = 0

            # Get load factor
            try:
                load_factor = self.df['load_factor'].quantile(0.95)
            except:
                load_factor = 1.0

            return {
                'min_descent_rate': min_descent,
                'max_climb_rate': max_climb,
                'climb_angle_max': climb_angle,
                'descent_angle': descent_angle,
                'stall_speed': stall_speed,
                'ld_max': ld_ratio,
                'predicted_takeoff_runway_length': predicted_takeoff_length,
                'predicted_landing_runway_length': predicted_landing_length,
                'actual_takeoff_runway_length': actual_takeoff_length,
                'actual_landing_runway_length': actual_landing_length,
                'load_factor_max': load_factor
            }
        except Exception as e:
            print(f"Error in get_performance_summary: {e}")
            # Return default values if something goes wrong
            return {
                'min_descent_rate': 0,
                'max_climb_rate': 0,
                'climb_angle_max': 0,
                'descent_angle': 0,
                'stall_speed': 0,
                'ld_max': 0,
                'predicted_takeoff_runway_length': 0,
                'predicted_landing_runway_length': 0,
                'actual_takeoff_runway_length': 0,
                'actual_landing_runway_length': 0,
                'load_factor_max': 1.0
            }