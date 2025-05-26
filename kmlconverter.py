import pandas as pd
import os
from pathlib import Path

def clean_coordinate(coord_str):
    """Convert coordinate string with multiple dots to proper decimal number"""
    try:
        # Remove any spaces
        coord_str = str(coord_str).strip()
        # Replace all dots except the last one with nothing
        parts = coord_str.split('.')
        if len(parts) > 2:
            return float(parts[0] + '.' + ''.join(parts[1:]))
        return float(coord_str)
    except:
        return None

def convert_to_kml(input_file, sheet_name=None, output_file='output.kml'):
    try:
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Read data based on file extension
        file_extension = os.path.splitext(input_file)[1].lower()
        try:
            if file_extension == '.csv':
                # Read CSV with semicolon separator
                df = pd.read_csv(input_file, sep=';')
                
                # Clean column names by removing leading/trailing spaces
                df.columns = df.columns.str.strip()
                
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(input_file, sheet_name=sheet_name)
            else:
                raise ValueError("Input file must be a CSV or Excel file (.csv, .xlsx, or .xls)")
                
        except PermissionError:
            raise PermissionError(f"Cannot read file. Make sure it's not open in another program: {input_file}")
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")
        
        # Define the exact column names from your CSV
        lat_col = 'Latitude[deg]'
        lon_col = 'Longitude[deg]'
        alt_col = 'Altitude[m]'
        time_col = 'Time[ms]'
        
        # Convert coordinates using the clean_coordinate function
        df[lat_col] = df[lat_col].apply(clean_coordinate)
        df[lon_col] = df[lon_col].apply(clean_coordinate)
        df[alt_col] = pd.to_numeric(df[alt_col], errors='coerce')
        
        # Apply altitude correction (similar to performancecalc.py)
        # Store original altitude
        df['original_altitude'] = df[alt_col].copy()
        
        # Calculate altitude correction based on start and end points
        flight_duration = (df[time_col].max() - df[time_col].min()) / 1000  # convert to seconds
        start_alt = df.loc[df[time_col] == df[time_col].min(), alt_col].iloc[0]
        end_alt = df.loc[df[time_col] == df[time_col].max(), alt_col].iloc[0]
        
        # Calculate linear correction
        altitude_offset = -(start_alt + end_alt) / 2
        
        print(f"Altitude offset correction: {altitude_offset:.2f}m (start: {start_alt:.2f}m, end: {end_alt:.2f}m)")
        
        # Apply correction
        df[alt_col] = df[alt_col] + altitude_offset
        
        # Convert milliseconds to seconds and group by second
        df['second'] = (df[time_col] / 1000).astype(int)
        df_sampled = df.groupby('second').first().reset_index()
        
        print(f"\nOriginal number of points: {len(df)}")
        print(f"Number of points after sampling: {len(df_sampled)}")
        
        # Print first few rows of coordinate data after conversion
        print("\nFirst few rows of coordinate data after conversion and sampling:")
        print(df_sampled[[lat_col, lon_col, alt_col]].head())
        
        # Remove rows with NaN values
        df_sampled = df_sampled.dropna(subset=[lat_col, lon_col, alt_col])
        
        if df_sampled.empty:
            raise ValueError("No valid coordinate data found after converting to numeric values")
            
        # Validate coordinate ranges
        if not all(-90 <= lat <= 90 for lat in df_sampled[lat_col]):
            raise ValueError("Latitude values must be between -90 and 90")
        if not all(-180 <= lon <= 180 for lon in df_sampled[lon_col]):
            raise ValueError("Longitude values must be between -180 and 180")
            
        # Write KML file
        with open(output_file, 'w') as kml:
            kml.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            kml.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
            kml.write('  <Document>\n')
            kml.write('    <name>GPS Path</name>\n')
            kml.write('    <description>Flightpath description</description>\n')
            kml.write('    <Style id="redLinelightredPoly">\n')
            kml.write('       <LineStyle>\n')
            kml.write('          <color>ff0000ff</color>\n')
            kml.write('          <width>2</width>\n')
            kml.write('       </LineStyle>\n')
            kml.write('       <PolyStyle>\n')
            kml.write('          <color>7f00007f</color>\n')
            kml.write('       </PolyStyle>\n')
            kml.write('    </Style>\n')
            kml.write('    <Placemark>\n')
            kml.write('      <name>Path</name>\n')
            kml.write('      <description>Transparent light red wall with red outlines</description>\n')
            kml.write('      <styleUrl>#redLinelightredPoly</styleUrl>\n')
            kml.write('      <LineString>\n')
            kml.write('        <extrude>1</extrude>\n')
            kml.write('        <tessellate>1</tessellate>\n')
            kml.write('        <altitudeMode>absolute</altitudeMode>\n')
            kml.write('        <coordinates>\n')

            # Write coordinates
            for _, row in df_sampled.iterrows():
                lat = row[lat_col]
                lon = row[lon_col]
                alt = row[alt_col]
                kml.write(f'          {lon},{lat},{alt}\n')

            # Close KML tags
            kml.write('        </coordinates>\n')
            kml.write('      </LineString>\n')
            kml.write('    </Placemark>\n')
            kml.write('  </Document>\n')
            kml.write('</kml>\n')
            
        print(f"\nKML file created: {output_file}")
        
    except Exception as e:
        raise Exception(f"Error processing file to KML: {str(e)}")

if __name__ == "__main__":
    while True:
        file_path = input("Enter full path to CSV or Excel file (e.g., C:\\Users\\You\\file.csv): ").strip()
        
        if not file_path:
            print("No file path entered. Please try again.")
            continue
            
        if not file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
            print("Please enter a valid CSV or Excel file path (ending with .csv, .xlsx, or .xls)")
            continue
            
        try:
            # Try saving to current directory instead of system folders
            output_file = os.path.join(os.path.dirname(__file__), 'output.kml')
            convert_to_kml(file_path, output_file=output_file)
            break
        except Exception as e:
            print(f"Error: {e}")
            retry = input("Would you like to try again? (y/n): ").lower()
            if retry != 'y':
                break