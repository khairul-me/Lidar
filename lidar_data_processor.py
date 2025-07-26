#!/usr/bin/env python3
"""
LiDAR Data Processor for LD06 and RPLidar C1
"""

import os
import sys
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime
import struct

class LidarDataProcessor:
    
    
    def __init__(self, data_path: str, output_dir: str = "processed_lidar_data"):
        
       
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # LiDAR specifications
        self.lidar_specs = {
            'ld06': {
                'fov': 360,  # Field of view in degrees
                'angular_resolution': 0.1,  # Angular resolution in degrees
                'max_range': 12.0,  # Maximum range in meters
                'min_range': 0.1,   # Minimum range in meters
            },
            'rplidar_a1': {
                'fov': 360,
                'angular_resolution': 1.0,
                'max_range': 12.0,
                'min_range': 0.15,
            }
        }
    
    def detect_data_format(self, file_path: Path) -> str:


        if file_path.suffix.lower() == '.db3':
            return 'ros2_bag'
        elif file_path.suffix.lower() == '.csv':
            return 'csv'
        elif file_path.suffix.lower() == '.json':
            return 'json'
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return 'excel'
        elif file_path.suffix.lower() in ['.bin', '.dat']:
            return 'binary'
        else:
            # Try to read as text and guess format
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{'):
                        return 'json'
                    elif ',' in first_line:
                        return 'csv'
                    else:
                        return 'unknown'
            except:
                return 'unknown'
    
    def read_ros2_bag_data(self, file_path: Path) -> List[Dict]:

        try:
            # Note: This requires rosbag2 library
            # pip install rosbag2
            from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
            
            storage_options = StorageOptions(uri=str(file_path), storage_id='sqlite3')
            converter_options = ConverterOptions('', '')
            reader = SequentialReader()
            reader.open(storage_options, converter_options)
            
            scan_data = []
            while reader.has_next():
                (topic_name, data, t) = reader.read_next()
                if 'scan' in topic_name.lower() or 'laser' in topic_name.lower():
                    # Parse LaserScan message
                    scan_data.append(self._parse_laserscan_message(data))
            
            return scan_data
            
        except ImportError:
            print("Warning: rosbag2 library not available. Install with: pip install rosbag2")
            return []
        except Exception as e:
            print(f"Error reading ROS2 bag: {e}")
            return []
    
    def read_csv_data(self, file_path: Path) -> List[Dict]:

        try:
            df = pd.read_csv(file_path)
            scan_data = []
            
            # Try to identify columns
            angle_cols = [col for col in df.columns if 'angle' in col.lower()]
            distance_cols = [col for col in df.columns if 'distance' in col.lower() or 'range' in col.lower()]
            
            if angle_cols and distance_cols:
                for _, row in df.iterrows():
                    scan_data.append({
                        'angles': row[angle_cols[0]].tolist() if isinstance(row[angle_cols[0]], (list, np.ndarray)) else [row[angle_cols[0]]],
                        'distances': row[distance_cols[0]].tolist() if isinstance(row[distance_cols[0]], (list, np.ndarray)) else [row[distance_cols[0]]],
                        'timestamp': row.get('timestamp', datetime.now().timestamp())
                    })
            else:
                # Assume first column is angles, second is distances
                for _, row in df.iterrows():
                    scan_data.append({
                        'angles': [row.iloc[0]],
                        'distances': [row.iloc[1]],
                        'timestamp': datetime.now().timestamp()
                    })
            
            return scan_data
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []
    
    def read_json_data(self, file_path: Path) -> List[Dict]:
 
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                print(f"Unexpected JSON format in {file_path}")
                return []
                
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return []
    
    def read_excel_data(self, file_path: Path) -> List[Dict]:
 
        try:
            df = pd.read_excel(file_path)
            scan_data = []
            
            # Try to identify columns
            angle_cols = [col for col in df.columns if 'angle' in col.lower()]
            distance_cols = [col for col in df.columns if 'distance' in col.lower() or 'range' in col.lower()]
            
            if angle_cols and distance_cols:
                for _, row in df.iterrows():
                    scan_data.append({
                        'angles': row[angle_cols[0]].tolist() if isinstance(row[angle_cols[0]], (list, np.ndarray)) else [row[angle_cols[0]]],
                        'distances': row[distance_cols[0]].tolist() if isinstance(row[distance_cols[0]], (list, np.ndarray)) else [row[distance_cols[0]]],
                        'timestamp': row.get('timestamp', datetime.now().timestamp())
                    })
            else:
                # Assume first column is angles, second is distances
                for _, row in df.iterrows():
                    scan_data.append({
                        'angles': [row.iloc[0]],
                        'distances': [row.iloc[1]],
                        'timestamp': datetime.now().timestamp()
                    })
            
            return scan_data
            
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return []
    
    def read_binary_data(self, file_path: Path, lidar_type: str = 'ld06') -> List[Dict]:
        """
        Read binary format LiDAR data
        
        Args:
            file_path: Path to the binary file
            lidar_type: Type of LiDAR ('ld06' or 'rplidar_a1')
            
        Returns:
            List of scan data dictionaries
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            scan_data = []
            
            if lidar_type == 'ld06':
                # LD06 binary format (example)
                # Each scan packet: 2 bytes header + 2 bytes length + scan data
                offset = 0
                while offset < len(data):
                    if offset + 4 > len(data):
                        break
                    
                    header = struct.unpack('<H', data[offset:offset+2])[0]
                    length = struct.unpack('<H', data[offset+2:offset+4])[0]
                    
                    if header == 0xAA55:  # LD06 header
                        scan_points = []
                        angles = []
                        distances = []
                        
                        for i in range(length):
                            if offset + 4 + i*3 > len(data):
                                break
                            point_data = struct.unpack('<H', data[offset+4+i*3:offset+6+i*3])[0]
                            distance = (point_data & 0x3FFF) / 1000.0  # Convert to meters
                            angle = (i * 0.1) % 360  # 0.1 degree resolution
                            
                            angles.append(angle)
                            distances.append(distance)
                        
                        scan_data.append({
                            'angles': angles,
                            'distances': distances,
                            'timestamp': datetime.now().timestamp()
                        })
                    
                    offset += 4 + length
            
            elif lidar_type == 'rplidar_a1':
                # RPLidar A1 binary format (example)
                # Each scan packet: 1 byte sync + 1 byte length + scan data
                offset = 0
                while offset < len(data):
                    if offset + 2 > len(data):
                        break
                    
                    sync = data[offset]
                    length = data[offset + 1]
                    
                    if sync == 0xA5:  # RPLidar sync byte
                        angles = []
                        distances = []
                        
                        for i in range(length):
                            if offset + 2 + i*2 > len(data):
                                break
                            point_data = struct.unpack('<H', data[offset+2+i*2:offset+4+i*2])[0]
                            distance = (point_data & 0x3FFF) / 1000.0
                            angle = (i * 1.0) % 360  # 1.0 degree resolution
                            
                            angles.append(angle)
                            distances.append(distance)
                        
                        scan_data.append({
                            'angles': angles,
                            'distances': distances,
                            'timestamp': datetime.now().timestamp()
                        })
                    
                    offset += 2 + length
            
            return scan_data
            
        except Exception as e:
            print(f"Error reading binary file: {e}")
            return []
    
    def _parse_laserscan_message(self, data: bytes) -> Dict:
        """
        Parse ROS2 LaserScan message data
        
        Args:
            data: Raw message data
            
        Returns:
            Parsed scan data dictionary
        """
        # This is a simplified parser - in practice you'd use ROS2 message types
        try:
            # Extract angles and ranges from LaserScan message
            # This is a placeholder - actual implementation depends on message format
            return {
                'angles': list(range(0, 360, 1)),  # 1-degree resolution
                'distances': [1.0] * 360,  # Placeholder distances
                'timestamp': datetime.now().timestamp()
            }
        except Exception as e:
            print(f"Error parsing LaserScan message: {e}")
            return {'angles': [], 'distances': [], 'timestamp': datetime.now().timestamp()}
    
    def process_scan_data(self, scan_data: List[Dict], lidar_type: str = 'ld06') -> Dict:
        """
        Process and analyze scan data
        
        Args:
            scan_data: List of scan data dictionaries
            lidar_type: Type of LiDAR sensor
            
        Returns:
            Processed data with statistics and analysis
        """
        if not scan_data:
            return {}
        
        all_angles = []
        all_distances = []
        
        for scan in scan_data:
            all_angles.extend(scan.get('angles', []))
            all_distances.extend(scan.get('distances', []))
        
        # Filter valid measurements
        valid_indices = [i for i, d in enumerate(all_distances) 
                        if self.lidar_specs[lidar_type]['min_range'] <= d <= self.lidar_specs[lidar_type]['max_range']]
        
        valid_angles = [all_angles[i] for i in valid_indices]
        valid_distances = [all_distances[i] for i in valid_indices]
        
        # Calculate statistics
        stats = {
            'total_scans': len(scan_data),
            'total_points': len(all_angles),
            'valid_points': len(valid_angles),
            'min_distance': min(valid_distances) if valid_distances else 0,
            'max_distance': max(valid_distances) if valid_distances else 0,
            'avg_distance': np.mean(valid_distances) if valid_distances else 0,
            'std_distance': np.std(valid_distances) if valid_distances else 0,
            'coverage_percentage': (len(valid_angles) / len(all_angles) * 100) if all_angles else 0
        }
        
        return {
            'angles': valid_angles,
            'distances': valid_distances,
            'statistics': stats,
            'lidar_type': lidar_type,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_human_readable_output(self, processed_data: Dict, filename: str = None):
        """
        Save processed data in human-readable format
        
        Args:
            processed_data: Processed LiDAR data
            filename: Output filename (optional)
        """
        if not processed_data:
            print("No data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lidar_scan_{timestamp}"
        
        # Save as CSV
        csv_path = self.output_dir / f"{filename}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Angle (degrees)', 'Distance (meters)', 'X (meters)', 'Y (meters)'])
            
            for angle, distance in zip(processed_data['angles'], processed_data['distances']):
                # Convert polar to Cartesian coordinates
                x = distance * np.cos(np.radians(angle))
                y = distance * np.sin(np.radians(angle))
                writer.writerow([f"{angle:.1f}", f"{distance:.3f}", f"{x:.3f}", f"{y:.3f}"])
        
        # Save statistics as JSON
        json_path = self.output_dir / f"{filename}_stats.json"
        with open(json_path, 'w') as f:
            json.dump(processed_data['statistics'], f, indent=2)
        
        # Save detailed report
        report_path = self.output_dir / f"{filename}_report.txt"
        with open(report_path, 'w') as f:
            f.write("LiDAR Data Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"LiDAR Type: {processed_data['lidar_type'].upper()}\n")
            f.write(f"Timestamp: {processed_data['timestamp']}\n\n")
            
            stats = processed_data['statistics']
            f.write("Scan Statistics:\n")
            f.write(f"  Total Scans: {stats['total_scans']}\n")
            f.write(f"  Total Points: {stats['total_points']}\n")
            f.write(f"  Valid Points: {stats['valid_points']}\n")
            f.write(f"  Coverage: {stats['coverage_percentage']:.1f}%\n\n")
            
            f.write("Distance Statistics:\n")
            f.write(f"  Minimum Distance: {stats['min_distance']:.3f} m\n")
            f.write(f"  Maximum Distance: {stats['max_distance']:.3f} m\n")
            f.write(f"  Average Distance: {stats['avg_distance']:.3f} m\n")
            f.write(f"  Standard Deviation: {stats['std_distance']:.3f} m\n\n")
            
            f.write("Sample Data Points:\n")
            f.write("Angle (deg) | Distance (m) | X (m) | Y (m)\n")
            f.write("-" * 45 + "\n")
            
            for i, (angle, distance) in enumerate(zip(processed_data['angles'][:20], processed_data['distances'][:20])):
                x = distance * np.cos(np.radians(angle))
                y = distance * np.sin(np.radians(angle))
                f.write(f"{angle:10.1f} | {distance:11.3f} | {x:5.3f} | {y:5.3f}\n")
            
            if len(processed_data['angles']) > 20:
                f.write(f"... and {len(processed_data['angles']) - 20} more points\n")
        
        print(f"Processed data saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  Statistics: {json_path}")
        print(f"  Report: {report_path}")
    
    def create_visualization(self, processed_data: Dict, filename: str = None):
        """
        Create visualization plots of the LiDAR data
        
        Args:
            processed_data: Processed LiDAR data
            filename: Output filename (optional)
        """
        if not processed_data:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lidar_plot_{timestamp}"
        
        # Create figure with polar and regular subplots
        fig = plt.figure(figsize=(15, 6))
        
        # Polar plot
        ax1 = fig.add_subplot(121, projection='polar')
        angles_rad = np.radians(processed_data['angles'])
        ax1.scatter(angles_rad, processed_data['distances'], alpha=0.6, s=1)
        ax1.set_theta_direction(-1)
        ax1.set_theta_zero_location('N')
        ax1.set_title(f'{processed_data["lidar_type"].upper()} - Polar View')
        ax1.grid(True)
        
        # Cartesian plot
        ax2 = fig.add_subplot(122)
        x_coords = [d * np.cos(np.radians(a)) for a, d in zip(processed_data['angles'], processed_data['distances'])]
        y_coords = [d * np.sin(np.radians(a)) for a, d in zip(processed_data['angles'], processed_data['distances'])]
        
        ax2.scatter(x_coords, y_coords, alpha=0.6, s=1)
        ax2.set_aspect('equal')
        ax2.set_title(f'{processed_data["lidar_type"].upper()} - Cartesian View')
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        ax2.grid(True)
        
        # Add origin point
        ax2.scatter(0, 0, color='red', s=100, marker='o', label='LiDAR Position')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {plot_path}")
    
    def process_file(self, file_path: Path, lidar_type: str = 'ld06') -> bool:
        """
        Process a single LiDAR data file
        
        Args:
            file_path: Path to the data file
            lidar_type: Type of LiDAR sensor
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Processing file: {file_path}")
        
        # Detect file format
        file_format = self.detect_data_format(file_path)
        print(f"Detected format: {file_format}")
        
        # Read data based on format
        scan_data = []
        if file_format == 'ros2_bag':
            scan_data = self.read_ros2_bag_data(file_path)
        elif file_format == 'csv':
            scan_data = self.read_csv_data(file_path)
        elif file_format == 'json':
            scan_data = self.read_json_data(file_path)
        elif file_format == 'excel':
            scan_data = self.read_excel_data(file_path)
        elif file_format == 'binary':
            scan_data = self.read_binary_data(file_path, lidar_type)
        else:
            print(f"Unsupported file format: {file_format}")
            return False
        
        if not scan_data:
            print("No data found in file")
            return False
        
        # Process the data
        processed_data = self.process_scan_data(scan_data, lidar_type)
        
        if not processed_data:
            print("Failed to process data")
            return False
        
        # Save output
        filename = f"{file_path.stem}_{lidar_type}"
        self.save_human_readable_output(processed_data, filename)
        self.create_visualization(processed_data, filename)
        
        return True
    
    def process_directory(self, directory_path: Path, lidar_type: str = 'ld06') -> int:
        """
        Process all LiDAR data files in a directory
        
        Args:
            directory_path: Path to the directory
            lidar_type: Type of LiDAR sensor
            
        Returns:
            Number of successfully processed files
        """
        if not directory_path.is_dir():
            print(f"Directory not found: {directory_path}")
            return 0
        
        processed_count = 0
        supported_extensions = ['.db3', '.csv', '.json', '.xlsx', '.xls', '.bin', '.dat']
        
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                if self.process_file(file_path, lidar_type):
                    processed_count += 1
        
        return processed_count


def main():
    """Main function to run the LiDAR data processor"""
    print("LiDAR Data Processor for LD06 and RPLidar A1")
    print("=" * 50)
    
    # Ask for data directory
    while True:
        data_path = input("\nEnter the directory path of your saved LiDAR data files: ").strip()
        
        # Remove quotes if user added them
        data_path = data_path.strip('"\'')
        
        if not data_path:
            print("Please enter a valid path.")
            continue
            
        data_path = Path(data_path)
        
        if not data_path.exists():
            print(f"Error: Path '{data_path}' does not exist. Please enter a valid path.")
            continue
            
        break
    
    # Ask for output directory
    output_dir = input("\nEnter output directory for processed data (or press Enter for default 'processed_lidar_data'): ").strip()
    if not output_dir:
        output_dir = "processed_lidar_data"
    
    # Ask for LiDAR type
    while True:
        lidar_type = input("\nSelect LiDAR type (1 for LD06, 2 for RPLidar A1, or press Enter for LD06): ").strip()
        
        if not lidar_type or lidar_type == "1":
            lidar_type = "ld06"
            break
        elif lidar_type == "2":
            lidar_type = "rplidar_a1"
            break
        else:
            print("Please enter 1, 2, or press Enter for LD06.")
    
    # Ask if user wants to process all files or specific files
    while True:
        process_mode = input("\nProcess mode:\n1. All files in directory\n2. Single file\nEnter choice (1 or 2): ").strip()
        
        if process_mode == "1":
            process_all = True
            break
        elif process_mode == "2":
            process_all = False
            break
        else:
            print("Please enter 1 or 2.")
    
    print(f"\nProcessing configuration:")
    print(f"  Data path: {data_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  LiDAR type: {lidar_type}")
    print(f"  Mode: {'All files' if process_all else 'Single file'}")
    
    # Confirm before processing
    confirm = input("\nProceed with processing? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Processing cancelled.")
        return
    
    # Create processor
    processor = LidarDataProcessor(str(data_path), output_dir)
    
    if process_all:
        # Process directory
        print(f"\nProcessing all files in: {data_path}")
        count = processor.process_directory(data_path, lidar_type)
        print(f"\n✓ Successfully processed {count} files")
    else:
        # List available files and let user choose
        supported_extensions = ['.db3', '.csv', '.json', '.xlsx', '.xls', '.bin', '.dat']
        available_files = [f for f in data_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in supported_extensions]
        
        if not available_files:
            print(f"\nNo supported LiDAR data files found in {data_path}")
            print("Supported formats: .db3, .csv, .json, .xlsx, .xls, .bin, .dat")
            return
        
        print(f"\nAvailable files in {data_path}:")
        for i, file_path in enumerate(available_files, 1):
            print(f"  {i}. {file_path.name}")
        
        while True:
            try:
                choice = input(f"\nSelect file to process (1-{len(available_files)}): ").strip()
                file_index = int(choice) - 1
                
                if 0 <= file_index < len(available_files):
                    selected_file = available_files[file_index]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_files)}.")
            except ValueError:
                print("Please enter a valid number.")
        
        print(f"\nProcessing file: {selected_file.name}")
        if processor.process_file(selected_file, lidar_type):
            print("\n✓ File processed successfully")
        else:
            print("\n✗ Failed to process file")
            sys.exit(1)
    
    print(f"\nOutput files saved to: {output_dir}")
    print("Generated files include:")
    print("  - CSV files with angles and distances")
    print("  - JSON files with statistics")
    print("  - Text reports with human-readable summaries")
    print("  - PNG files with visualizations")


if __name__ == "__main__":
    main() 