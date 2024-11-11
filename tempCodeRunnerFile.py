import torch
from PIL import Image
import torchvision.transforms as T
import torchvision
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from matplotlib.patches import Polygon
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
import cv2

class ParkingSpotDetector:
    def __init__(self):
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])
        
        # COCO dataset vehicle-related classes
        self.vehicle_classes = {
            2: 'bicycle',
            3: 'car',
            4: 'motorcycle',
            6: 'bus',
            8: 'truck'
        }
        
    def detect_cars(self, image_path):
        """Detect all types of vehicles in the image and return their masks and boxes"""
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # Get predictions
        masks = predictions[0]['masks']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']
        boxes = predictions[0]['boxes']
        
        # Get all vehicle detections with confidence threshold
        vehicle_detections = []
        for i in range(len(labels)):
            label = labels[i].item()
            if label in self.vehicle_classes and scores[i] > 0.15:
                vehicle_detections.append({
                    'mask': masks[i].squeeze().numpy() > 0.5,
                    'box': boxes[i].numpy(),
                    'score': scores[i].item(),
                    'class': self.vehicle_classes[label]
                })
        
        return vehicle_detections, np.array(img)

    def calculate_overlap(self, spot_coords, car_box):
        """Calculate overlap ratio between a parking spot and a car box"""
        spot_coords = np.array(spot_coords)
        spot_x_min, spot_y_min = np.min(spot_coords, axis=0)
        spot_x_max, spot_y_max = np.max(spot_coords, axis=0)
        spot_area = (spot_x_max - spot_x_min) * (spot_y_max - spot_y_min)
        
        # Calculate box overlap
        x_left = max(spot_x_min, car_box[0])
        y_top = max(spot_y_min, car_box[1])
        x_right = min(spot_x_max, car_box[2])
        y_bottom = min(spot_y_max, car_box[3])
        
        if x_right > x_left and y_bottom > y_top:
            overlap_area = (x_right - x_left) * (y_bottom - y_top)
            return overlap_area / spot_area
        return 0

    def assign_cars_to_spots(self, parking_spots, car_detections):
        """Assign each car to at most one parking spot based on maximum overlap, 
        strictly prioritizing lower spots for each car"""
        # Create a matrix of overlap ratios between all spots and cars
        overlap_matrix = []
        spot_y_coords = []  # Store Y-coordinates for each spot
        
        for spot in parking_spots:
            spot_overlaps = []
            # Calculate average Y coordinate for the spot
            avg_y = np.mean([coord[1] for coord in spot])
            spot_y_coords.append(avg_y)
            
            for car in car_detections:
                overlap_ratio = self.calculate_overlap(spot, car['box'])
                spot_overlaps.append(overlap_ratio)
            overlap_matrix.append(spot_overlaps)
        
        overlap_matrix = np.array(overlap_matrix)
        spot_y_coords = np.array(spot_y_coords)
        
        # Track which spots are occupied and which cars are assigned
        occupied_spots = set()
        assigned_cars = set()
        spot_car_assignments = {}  # Maps spot index to car index
        
        # Process each car
        for car_idx in range(len(car_detections)):
            if car_idx in assigned_cars:
                continue
                
            # Get all spots that this car overlaps with sufficiently
            car_overlaps = overlap_matrix[:, car_idx]
            valid_spots = np.where(car_overlaps >= 0.3)[0]
            
            if len(valid_spots) == 0:
                continue
                
            # Among valid spots, find the one with highest Y coordinate (lowest in image)
            # that isn't already occupied
            valid_spots_y = [(spot_idx, spot_y_coords[spot_idx]) 
                            for spot_idx in valid_spots 
                            if spot_idx not in occupied_spots]
            
            if not valid_spots_y:
                continue
                
            # Sort by Y coordinate (highest Y = lowest in image)
            valid_spots_y.sort(key=lambda x: x[1], reverse=True)
            
            # Assign car to the lowest available spot
            best_spot_idx = valid_spots_y[0][0]
            occupied_spots.add(best_spot_idx)
            assigned_cars.add(car_idx)
            spot_car_assignments[best_spot_idx] = car_idx
        
        return occupied_spots, spot_car_assignments

    def visualize_results(self, image_path, parking_spots):
        """Visualize parking spot occupancy with colored overlays"""
        # Detect cars
        car_detections, img_np = self.detect_cars(image_path)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        plt.imshow(img_np)
        
        # Assign cars to spots
        occupied_spot_indices, spot_car_assignments = self.assign_cars_to_spots(parking_spots, car_detections)
        
        # Plot all spots
        for i, spot in enumerate(parking_spots):
            if i in occupied_spot_indices:
                color = 'red'
            else:
                color = 'green'
            polygon = Polygon(spot, color=color, alpha=0.4)
            plt.gca().add_patch(polygon)
        
        # Add legend
        handles = [
            plt.Rectangle((0,0), 1, 1, color='red', alpha=0.4),
            plt.Rectangle((0,0), 1, 1, color='green', alpha=0.4)
        ]
        labels = ['Occupied', 'Vacant']
        plt.legend(handles, labels, loc='upper right')
        
        plt.title("Parking Spot Occupancy Detection")
        plt.axis('off')
        plt.show()
        
        return {
            'total_spots': len(parking_spots),
            'occupied_spots': len(occupied_spot_indices),
            'vacant_spots': len(parking_spots) - len(occupied_spot_indices),
            'occupancy_rate': (len(occupied_spot_indices) / len(parking_spots)) * 100
        }

# Example usage:
if __name__ == "__main__":
    detector = ParkingSpotDetector()

    # Load parking spot regions
    with open('regions/AAlotWestRegions3.p', 'rb') as f:
        parking_spots = pickle.load(f)

    # Analyze and visualize parking lot
    results = detector.visualize_results('images/AA lot West 1.jpg', parking_spots)

    print(f"\nParking Lot Analysis:")
    print(f"Total spots: {results['total_spots']}")
    print(f"Occupied spots: {results['occupied_spots']}")
    print(f"Vacant spots: {results['vacant_spots']}")
    print(f"Occupancy rate: {results['occupancy_rate']:.1f}%")