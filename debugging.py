import torch
from PIL import Image
import torchvision.transforms as T
import torchvision
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from matplotlib.patches import Polygon, Rectangle
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
import cv2

class ParkingSpotDetector:
    def __init__(self):
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])
        
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
        
        masks = predictions[0]['masks']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']
        boxes = predictions[0]['boxes']
        
        # Debug print
        print("\nInitial detections:")
        for i in range(len(labels)):
            label = labels[i].item()
            score = scores[i].item()
            if label in self.vehicle_classes:
                print(f"Detected {self.vehicle_classes[label]} with score {score:.3f}")
        
        vehicle_detections = []
        detected_cars = []

        for i in range(len(labels)):
            label = labels[i].item()
            score = scores[i].item()
            
            if label in self.vehicle_classes and score > 0.15:
                mask = masks[i].squeeze().numpy() > 0.5
                box = boxes[i].numpy()
                vehicle_class = self.vehicle_classes[label]

                if vehicle_class == 'truck' and any(self.calculate_overlap(car['box'], box) > 0.5 for car in detected_cars):
                    print(f"Skipping truck detection due to overlap with existing car")
                    continue

                if vehicle_class == 'car':
                    detected_cars.append({'box': box, 'mask': mask})

                vehicle_detections.append({
                    'mask': mask,
                    'box': box,
                    'score': score,
                    'class': vehicle_class
                })

        print(f"\nFinal vehicle detections after filtering: {len(vehicle_detections)}")
        for i, det in enumerate(vehicle_detections):
            print(f"Vehicle {i}: {det['class']} (score: {det['score']:.3f}) at box {det['box']}")
        
        return vehicle_detections, np.array(img)

    def calculate_overlap(self, spot, car_box):
        """Calculate overlap ratio between a parking spot and a car box"""
        if isinstance(spot, np.ndarray) and spot.shape != (4,):
            x_min, y_min = np.min(spot, axis=0)
            x_max, y_max = np.max(spot, axis=0)
            spot = [x_min, y_min, x_max, y_max]

        x1, y1, x2, y2 = spot
        x1_p, y1_p, x2_p, y2_p = car_box

        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)
        
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            spot_area = (x2 - x1) * (y2 - y1)
            overlap_ratio = inter_area / spot_area
            return overlap_ratio
        return 0

    def assign_cars_to_spots(self, parking_spots, car_detections):
        """Assign each car to at most one parking spot based on maximum overlap"""
        overlap_matrix = []
        spot_y_coords = []
        
        print("\nCalculating overlaps:")
        for spot_idx, spot in enumerate(parking_spots):
            spot_overlaps = []
            avg_y = np.mean([coord[1] for coord in spot])
            spot_y_coords.append(avg_y)
            
            for car_idx, car in enumerate(car_detections):
                overlap_ratio = self.calculate_overlap(spot, car['box'])
                if overlap_ratio > 0:
                    print(f"Spot {spot_idx} overlaps with {car['class']} {car_idx} by {overlap_ratio:.3f}")
                spot_overlaps.append(overlap_ratio)
            overlap_matrix.append(spot_overlaps)
        
        overlap_matrix = np.array(overlap_matrix)
        spot_y_coords = np.array(spot_y_coords)
        
        occupied_spots = set()
        assigned_cars = set()
        spot_car_assignments = {}
        
        print("\nAssigning cars to spots:")
        for car_idx in range(len(car_detections)):
            if car_idx in assigned_cars:
                continue
                
            car_overlaps = overlap_matrix[:, car_idx]
            valid_spots = np.where(car_overlaps >= 0.3)[0]
            
            print(f"\nProcessing {car_detections[car_idx]['class']} {car_idx}:")
            print(f"Valid spots with >0.3 overlap: {valid_spots}")
            
            if len(valid_spots) == 0:
                print(f"No valid spots found for vehicle {car_idx}")
                continue
                
            valid_spots_y = [(spot_idx, spot_y_coords[spot_idx]) 
                            for spot_idx in valid_spots 
                            if spot_idx not in occupied_spots]
            
            if not valid_spots_y:
                print(f"All valid spots already occupied")
                continue
                
            valid_spots_y.sort(key=lambda x: x[1], reverse=True)
            
            best_spot_idx = valid_spots_y[0][0]
            occupied_spots.add(best_spot_idx)
            assigned_cars.add(car_idx)
            spot_car_assignments[best_spot_idx] = car_idx
            print(f"Assigned to spot {best_spot_idx}")
        
        return occupied_spots, spot_car_assignments

    def visualize_results(self, image_path, parking_spots):
        """Visualize parking spot occupancy with enhanced debugging visualization"""
        car_detections, img_np = self.detect_cars(image_path)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(img_np)
        
        occupied_spot_indices, spot_car_assignments = self.assign_cars_to_spots(parking_spots, car_detections)
        
        # Plot all spots
        for i, spot in enumerate(parking_spots):
            if i in occupied_spot_indices:
                color = 'red'
                alpha = 0.4
            else:
                color = 'green'
                alpha = 0.4
            polygon = Polygon(spot, color=color, alpha=alpha)
            plt.gca().add_patch(polygon)
            
            # Add spot number for debugging
            centroid = np.mean(spot, axis=0)
            plt.text(centroid[0], centroid[1], f'Spot {i}', 
                    color='white', fontsize=8, ha='center', va='center')
        
        # Plot car detection boxes with their indices
        for i, car in enumerate(car_detections):
            box = car['box']
            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                           fill=False, color='yellow', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(box[0], box[1], f"{car['class']} {i}\n{car['score']:.2f}", 
                    color='yellow', fontsize=8)
        
        handles = [
            plt.Rectangle((0,0), 1, 1, color='red', alpha=0.4),
            plt.Rectangle((0,0), 1, 1, color='green', alpha=0.4),
            plt.Rectangle((0,0), 1, 1, color='yellow', fill=False)
        ]
        labels = ['Occupied', 'Vacant', 'Detected Vehicle']
        plt.legend(handles, labels, loc='upper right')
        
        plt.title("Parking Spot Occupancy Detection (Debug View)")
        plt.axis('off')
        plt.show()
        
        results = {
            'total_spots': len(parking_spots),
            'occupied_spots': len(occupied_spot_indices),
            'vacant_spots': len(parking_spots) - len(occupied_spot_indices),
            'occupancy_rate': (len(occupied_spot_indices) / len(parking_spots)) * 100
        }
        
        print("\nDetailed Results:")
        print(f"Total spots: {results['total_spots']}")
        print(f"Occupied spots: {results['occupied_spots']}")
        print(f"Vacant spots: {results['vacant_spots']}")
        print(f"Occupancy rate: {results['occupancy_rate']:.1f}%")
        
        print("\nSpot-Car Assignments:")
        for spot_idx, car_idx in spot_car_assignments.items():
            car = car_detections[car_idx]
            print(f"Spot {spot_idx} -> {car['class']} {car_idx} (score: {car['score']:.3f})")
        
        return results

# Example usage:
if __name__ == "__main__":
    detector = ParkingSpotDetector()
    
    with open('regions/AAlotWestRegions3.p', 'rb') as f:
        parking_spots = pickle.load(f)
    
    results = detector.visualize_results('images/AA lot West 1.jpg', parking_spots)