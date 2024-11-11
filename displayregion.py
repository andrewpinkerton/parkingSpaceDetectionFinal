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
        
        self.vehicle_classes = {
            3: 'car',
            8: 'truck'
        }
    
    def calculate_box_overlap(self, box1, box2, class1, class2):
        """Calculate IoU between two bounding boxes, but only if one is a car and one is a truck"""
        # Check if we have one car and one truck
        if not ((class1 == 'car' and class2 == 'truck') or 
                (class1 == 'truck' and class2 == 'car')):
            return 0.0
            
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union
        
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
        
        # Create initial detections list with scores for sorting
        initial_detections = []
        for i in range(len(labels)):
            label = labels[i].item()
            score = scores[i].item()
            
            if label in self.vehicle_classes and score > 0.15:
                initial_detections.append({
                    'mask': masks[i].squeeze().numpy() > 0.5,
                    'box': boxes[i].numpy(),
                    'score': score,
                    'class': self.vehicle_classes[label]
                })
        
        # Sort detections by confidence score
        initial_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter out overlapping detections
        final_detections = []
        used_boxes = []
        used_classes = []  # Keep track of classes along with boxes
        
        for detection in initial_detections:
            # Check if this detection overlaps significantly with any existing detection
            is_overlap = False
            for idx, used_box in enumerate(used_boxes):
                if self.calculate_box_overlap(detection['box'], used_box, 
                                           detection['class'], used_classes[idx]) > 0.3:  # IoU threshold
                    is_overlap = True
                    break
            
            if not is_overlap:
                final_detections.append(detection)
                used_boxes.append(detection['box'])
                used_classes.append(detection['class'])
        
        return final_detections, np.array(img)

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
        
        for spot in parking_spots:
            spot_overlaps = []
            avg_y = np.mean([coord[1] for coord in spot])
            spot_y_coords.append(avg_y)
            
            for car in car_detections:
                overlap_ratio = self.calculate_overlap(spot, car['box'])
                spot_overlaps.append(overlap_ratio)
            overlap_matrix.append(spot_overlaps)
        
        overlap_matrix = np.array(overlap_matrix)
        spot_y_coords = np.array(spot_y_coords)
        
        occupied_spots = set()
        assigned_cars = set()
        spot_car_assignments = {}
        
        for car_idx in range(len(car_detections)):
            if car_idx in assigned_cars:
                continue
                
            car_overlaps = overlap_matrix[:, car_idx]
            valid_spots = np.where(car_overlaps >= 0.3)[0]
            
            if len(valid_spots) == 0:
                continue
                
            valid_spots_y = [(spot_idx, spot_y_coords[spot_idx]) 
                            for spot_idx in valid_spots 
                            if spot_idx not in occupied_spots]
            
            if not valid_spots_y:
                continue
                
            valid_spots_y.sort(key=lambda x: x[1], reverse=True)
            
            best_spot_idx = valid_spots_y[0][0]
            occupied_spots.add(best_spot_idx)
            assigned_cars.add(car_idx)
            spot_car_assignments[best_spot_idx] = car_idx
        
        return occupied_spots, spot_car_assignments

    def visualize_results(self, image_path, parking_spots):
        """Visualize parking spot occupancy with colored overlays"""
        car_detections, img_np = self.detect_cars(image_path)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(img_np)
        
        occupied_spot_indices, spot_car_assignments = self.assign_cars_to_spots(parking_spots, car_detections)
        
        for i, spot in enumerate(parking_spots):
            if i in occupied_spot_indices:
                color = 'red'
            else:
                color = 'green'
            polygon = Polygon(spot, color=color, alpha=0.4)
            plt.gca().add_patch(polygon)
        
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

if __name__ == "__main__":
    detector = ParkingSpotDetector()
    
    with open('regions/AAlotWestRegions3.p', 'rb') as f:
        parking_spots = pickle.load(f)
    
    results = detector.visualize_results('images/AA lot West 3.jpg', parking_spots)