import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
import sys


# read in avi video as tensor 
def avi2video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- Preprocessing Steps ---
        # 1. Convert to Grayscale (echos are inherently grayscale)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize to a standard size (the original EchoNet paper used 112x112)
        frame = cv2.resize(frame, (112, 112))
        
        frames.append(frame)

    cap.release()
    video_tensor = np.stack(frames, axis=0) # Shape: (T, H, W)
    return video_tensor

# get ed and es dataframe for file name
def extract_ed_es_tracings(df_tracings, file_name):
    subject_tracings = df_tracings[df_tracings['FileName'] == file_name]
    
    frames_ed, frames_es = subject_tracings['Frame'].unique()[0], subject_tracings['Frame'].unique()[1]

    df_ed = subject_tracings[subject_tracings['Frame'] == frames_ed]
    df_es = subject_tracings[subject_tracings['Frame'] == frames_es]
    
    return df_ed, df_es

# visualize long and short axis from coordinates 
def plot_long_short_axis(one_phase_df):

    # extract long axis from first entry according to documentation of echonet
    long_axis_row = one_phase_df.iloc[0]
    apex = (long_axis_row['X1'], long_axis_row['Y1'])
    base = (long_axis_row['X2'], long_axis_row['Y2'])

    # Subsequent rows are the short axes
    short_axis_df = one_phase_df.iloc[1:]
    side1_points = short_axis_df[['X1', 'Y1']].values
    side2_points = short_axis_df[['X2', 'Y2']].values

    plt.figure(figsize=(5, 5))
    plt.title(f'Visualization of Tracings')

    # Plot the long axis (Apex to Base)
    plt.plot([apex[0], base[0]], [apex[1], base[1]], 'r-', label='Long Axis', linewidth=2)
    plt.plot(apex[0], apex[1], 'ro', markersize=8, label='Apex')
    plt.plot(base[0], base[1], 'rx', markersize=8, label='Base')

    # Plot the short axis chords
    for i in range(len(side1_points)):
        p1 = side1_points[i]
        p2 = side2_points[i]
        # Use a different label only for the first one to avoid clutter
        label = 'Short Axes' if i == 0 else ''
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.6, label=label)

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis() # In image coordinates, Y typically goes down
    plt.axis('equal')
    plt.show()


# get binary mask from tracings 
def binary_mask_from_tracings(one_phase_df):

    long_axis_row = one_phase_df.iloc[0]
    apex = (long_axis_row['X1'], long_axis_row['Y1'])

    short_axis_df = one_phase_df.iloc[1:]
    side1_points = short_axis_df[['X1', 'Y1']].values
    side2_points = short_axis_df[['X2', 'Y2']].values

    contour_points = np.concatenate([
        [apex],              # Start at the apex
        side1_points,        # Go down one side
        side2_points[::-1]   # Go up the other side (in reverse)
    ]).astype(np.int32)

    height, width = 112, 112
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(mask, [contour_points], 1) # Use 1 for the mask value (or 255)

    return mask


# calculating the volume from the tracings
def calculate_volume_from_tracings(frame_df):
    """Calculates the LV volume using the Single-Plane Simpson's Method of Disks."""
    
    if frame_df.shape[0] < 2:
        # Not enough points to calculate volume
        return 0

    # 1. Calculate the Long Axis Length (L)
    long_axis_row = frame_df.iloc[0]
    apex = (long_axis_row['X1'], long_axis_row['Y1'])
    base = (long_axis_row['X2'], long_axis_row['Y2'])
    L = np.sqrt((base[0] - apex[0])**2 + (base[1] - apex[1])**2)
    
    # 2. Get the number of disks (N)
    short_axis_df = frame_df.iloc[1:]
    N = short_axis_df.shape[0]
    if N == 0:
        return 0

    # 3. Calculate the sum of squared diameters (Σ D_i²)
    sum_D_squared = 0
    for _, row in short_axis_df.iterrows():
        p1 = (row['X1'], row['Y1'])
        p2 = (row['X2'], row['Y2'])
        D_i = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        sum_D_squared += D_i**2
        
    # 4. Calculate the total volume (V)
    # V = (π * L) / (4 * N) * Σ(D_i²)
    volume = (np.pi * L) / (4 * N) * sum_D_squared
    
    return volume




########### traces estimation from mask gere
def line_segment_intersection(p1, p2, p3, p4):
    """
    Find intersection point of two line segments if it exists.
    
    Parameters:
    -----------
    p1, p2 : numpy arrays
        Start and end points of first line segment
    p3, p4 : numpy arrays
        Start and end points of second line segment
    
    Returns:
    --------
    numpy array of intersection point or None if no intersection
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return np.array([x, y])
    
    return None

def find_line_contour_intersections(line_start, line_end, contour_points, 
                                   reference_point, tolerance=5):
    """
    Find intersections between a line and a contour.
    
    Parameters:
    -----------
    line_start, line_end : numpy arrays
        Start and end points of the line
    contour_points : numpy array
        Points defining the contour
    reference_point : numpy array
        Reference point for finding nearby intersections
    tolerance : float
        Distance tolerance for finding intersections
    
    Returns:
    --------
    numpy array of intersection points
    """
    intersections = []
    
    # Check each segment of the contour
    for i in range(len(contour_points)):
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % len(contour_points)]
        
        # Find intersection between two line segments
        intersection = line_segment_intersection(line_start, line_end, p1, p2)
        
        if intersection is not None:
            # Check if this intersection is not too close to existing ones
            is_new = True
            for existing in intersections:
                if np.linalg.norm(existing - intersection) < tolerance:
                    is_new = False
                    break
            
            if is_new:
                intersections.append(intersection)
    
    return np.array(intersections) if intersections else np.array([])




def extract_lv_axes(binary_mask, num_short_axes=20):
    """
    Extract long axis (apex to base) and short axes from a binary mask of the left ventricle.
    
    Parameters:
    -----------
    binary_mask : numpy.ndarray
        Binary mask of the left ventricle (2D array with values 0 or 1)
    num_short_axes : int
        Number of short axes to generate along the long axis
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns X1, Y1, X2, Y2 where:
        - First row: apex (X1,Y1) and base (X2,Y2) coordinates
        - Subsequent rows: endpoints of short axes
    """
    
    # Ensure binary mask is uint8
    mask = (binary_mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        raise ValueError("No contours found in the binary mask")
    
    # Get the largest contour (in case there are multiple)
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour.squeeze()
    
    # If contour has wrong shape, reshape it
    if len(contour_points.shape) == 3:
        contour_points = contour_points.squeeze()
    
    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        raise ValueError("Cannot compute centroid - zero area")
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])
    
    # Find apex: furthest point from centroid
    distances_from_centroid = np.sqrt(np.sum((contour_points - centroid)**2, axis=1))
    apex_idx = np.argmax(distances_from_centroid)
    apex = contour_points[apex_idx]
    
    # Find base: extend line from apex through centroid to find intersection on opposite side
    # Vector from apex to centroid
    apex_to_centroid = centroid - apex
    apex_to_centroid_unit = apex_to_centroid / np.linalg.norm(apex_to_centroid)
    
    # Extend the line beyond centroid to find intersections with contour
    # We need to find where the line from apex through centroid intersects the contour
    # on the opposite side of the centroid
    
    # Create a long line from apex through centroid and beyond
    line_length = max(mask.shape) * 2
    line_start = apex
    line_end = apex + apex_to_centroid_unit * line_length
    
    # Find all intersections of this line with the contour
    # We'll need to check each contour segment
    intersections = []
    for i in range(len(contour_points)):
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % len(contour_points)]
        
        # Check if line segment intersects with contour segment
        intersection = line_segment_intersection(line_start, line_end, p1, p2)
        if intersection is not None:
            # Check if intersection is on the opposite side of centroid from apex
            vec_to_intersection = intersection - centroid
            vec_to_apex = apex - centroid
            
            # If dot product is negative, they're on opposite sides
            if np.dot(vec_to_intersection, vec_to_apex) < 0:
                intersections.append(intersection)
    
    if len(intersections) == 0:
        # Fallback: find the contour point closest to the extended line on the opposite side
        best_dist = float('inf')
        base = None
        
        for point in contour_points:
            # Check if point is on opposite side of centroid from apex
            if np.dot(point - centroid, apex - centroid) < 0:
                # Calculate distance from point to the line
                # Distance from point to line defined by apex and centroid
                line_vec = apex_to_centroid
                point_vec = point - apex
                
                # Project point onto line
                projection_length = np.dot(point_vec, apex_to_centroid_unit)
                projection = apex + apex_to_centroid_unit * projection_length
                
                # Distance is the distance from point to its projection
                dist = np.linalg.norm(point - projection)
                
                if dist < best_dist:
                    best_dist = dist
                    base = point
    else:
        # Among all intersections on the opposite side, choose the one furthest from apex
        distances = [np.linalg.norm(pt - apex) for pt in intersections]
        base = intersections[np.argmax(distances)]
    
    # Rest of the code remains the same...
    # Create the long axis vector
    long_axis_vector = base - apex
    long_axis_length = np.linalg.norm(long_axis_vector)
    long_axis_unit = long_axis_vector / long_axis_length
    
    # Generate short axes perpendicular to the long axis
    short_axes = []
    
    # Create perpendicular vector (rotate by 90 degrees)
    perpendicular = np.array([-long_axis_unit[1], long_axis_unit[0]])
    
    # Sample points along the long axis (excluding apex and base)
    for i in range(1, num_short_axes + 1):
        # Position along the long axis (from 0 to 1)
        t = i / (num_short_axes + 1)
        
        # Point on the long axis
        point_on_axis = apex + t * long_axis_vector
        
        # Find intersections of perpendicular line with contour
        # Create a long perpendicular line
        line_length = max(mask.shape) * 2
        line_start = point_on_axis - perpendicular * line_length
        line_end = point_on_axis + perpendicular * line_length
        
        # Find intersections with the contour
        intersections = find_line_contour_intersections(
            line_start, line_end, contour_points, point_on_axis
        )
        
        if len(intersections) >= 2:
            # Get the two intersection points closest to the axis point
            distances = np.sqrt(np.sum((intersections - point_on_axis)**2, axis=1))
            
            # Separate points on each side of the long axis
            side1_points = []
            side2_points = []
            
            for inter_point in intersections:
                # Determine which side of the long axis this point is on
                cross_product = np.cross(long_axis_vector, inter_point - apex)
                if cross_product > 0:
                    side1_points.append(inter_point)
                else:
                    side2_points.append(inter_point)
            
            # Get closest point from each side
            if len(side1_points) > 0 and len(side2_points) > 0:
                side1_point = min(side1_points, 
                                key=lambda p: np.linalg.norm(p - point_on_axis))
                side2_point = min(side2_points, 
                                key=lambda p: np.linalg.norm(p - point_on_axis))
                short_axes.append([side1_point[0], side1_point[1], 
                                 side2_point[0], side2_point[1]])
    
    # Create DataFrame
    data = []
    
    # First row: apex and base
    data.append({
        'X1': float(apex[0]),
        'Y1': float(apex[1]),
        'X2': float(base[0]),
        'Y2': float(base[1])
    })
    
    # Subsequent rows: short axes
    for axis_coords in short_axes:
        data.append({
            'X1': float(axis_coords[0]),
            'Y1': float(axis_coords[1]),
            'X2': float(axis_coords[2]),
            'Y2': float(axis_coords[3])
        })
    
    return pd.DataFrame(data)


# def extract_lv_axes(binary_mask, num_short_axes=20):
#     """
#     Extract long axis (apex to base) and short axes from a binary mask of the left ventricle.
    
#     Parameters:
#     -----------
#     binary_mask : numpy.ndarray
#         Binary mask of the left ventricle (2D array with values 0 or 1)
#     num_short_axes : int
#         Number of short axes to generate along the long axis
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with columns X1, Y1, X2, Y2 where:
#         - First row: apex (X1,Y1) and base (X2,Y2) coordinates
#         - Subsequent rows: endpoints of short axes
#     """
    
#     # Ensure binary mask is uint8
#     mask = (binary_mask > 0).astype(np.uint8)
    
#     # Find contours
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
#     if len(contours) == 0:
#         raise ValueError("No contours found in the binary mask")
    
#     # Get the largest contour (in case there are multiple)
#     contour = max(contours, key=cv2.contourArea)
#     contour_points = contour.squeeze()
    
#     # If contour has wrong shape, reshape it
#     if len(contour_points.shape) == 3:
#         contour_points = contour_points.squeeze()
    
#     # Calculate centroid
#     M = cv2.moments(contour)
#     if M["m00"] == 0:
#         raise ValueError("Cannot compute centroid - zero area")
    
#     cx = int(M["m10"] / M["m00"])
#     cy = int(M["m01"] / M["m00"])
#     centroid = np.array([cx, cy])
    
#     # Find apex: furthest point from centroid
#     distances_from_centroid = np.sqrt(np.sum((contour_points - centroid)**2, axis=1))
#     apex_idx = np.argmax(distances_from_centroid)
#     apex = contour_points[apex_idx]
    
#     # Find base: look for points on the opposite side of the centroid from apex
#     # Vector from centroid to apex
#     apex_vector = apex - centroid
    
#     # For each contour point, calculate the angle with the apex vector
#     angles = []
#     for point in contour_points:
#         point_vector = point - centroid
#         # Calculate angle using dot product
#         cos_angle = np.dot(apex_vector, point_vector) / (
#             np.linalg.norm(apex_vector) * np.linalg.norm(point_vector) + 1e-10
#         )
#         angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
    
#     angles = np.array(angles)
    
#     # Base should be roughly opposite to apex (angle close to π)
#     # Look for points with angles between 2π/3 and π
#     opposite_mask = (angles > 2*np.pi/3) & (angles <= np.pi)
    
#     if np.sum(opposite_mask) == 0:
#         # If no points found in that range, take the point with maximum angle
#         base_idx = np.argmax(angles)
#     else:
#         # Among opposite points, find the one that creates the longest axis
#         opposite_indices = np.where(opposite_mask)[0]
#         distances_to_apex = np.sqrt(np.sum((contour_points[opposite_indices] - apex)**2, axis=1))
#         base_idx = opposite_indices[np.argmax(distances_to_apex)]
    
#     base = contour_points[base_idx]
    
#     # Create the long axis vector
#     long_axis_vector = base - apex
#     long_axis_length = np.linalg.norm(long_axis_vector)
#     long_axis_unit = long_axis_vector / long_axis_length
    
#     # Generate short axes perpendicular to the long axis
#     short_axes = []
    
#     # Create perpendicular vector (rotate by 90 degrees)
#     perpendicular = np.array([-long_axis_unit[1], long_axis_unit[0]])
    
#     # Sample points along the long axis (excluding apex and base)
#     for i in range(1, num_short_axes + 1):
#         # Position along the long axis (from 0 to 1)
#         t = i / (num_short_axes + 1)
        
#         # Point on the long axis
#         point_on_axis = apex + t * long_axis_vector
        
#         # Find intersections of perpendicular line with contour
#         # Create a long perpendicular line
#         line_length = max(mask.shape) * 2
#         line_start = point_on_axis - perpendicular * line_length
#         line_end = point_on_axis + perpendicular * line_length
        
#         # Find intersections with the contour
#         intersections = find_line_contour_intersections(
#             line_start, line_end, contour_points, point_on_axis
#         )
        
#         if len(intersections) >= 2:
#             # Get the two intersection points closest to the axis point
#             distances = np.sqrt(np.sum((intersections - point_on_axis)**2, axis=1))
            
#             # Separate points on each side of the long axis
#             side1_points = []
#             side2_points = []
            
#             for inter_point in intersections:
#                 # Determine which side of the long axis this point is on
#                 cross_product = np.cross(long_axis_vector, inter_point - apex)
#                 if cross_product > 0:
#                     side1_points.append(inter_point)
#                 else:
#                     side2_points.append(inter_point)
            
#             # Get closest point from each side
#             if len(side1_points) > 0 and len(side2_points) > 0:
#                 side1_point = min(side1_points, 
#                                 key=lambda p: np.linalg.norm(p - point_on_axis))
#                 side2_point = min(side2_points, 
#                                 key=lambda p: np.linalg.norm(p - point_on_axis))
#                 short_axes.append([side1_point[0], side1_point[1], 
#                                  side2_point[0], side2_point[1]])
    
#     # Create DataFrame
#     data = []
    
#     # First row: apex and base
#     data.append({
#         'X1': float(apex[0]),
#         'Y1': float(apex[1]),
#         'X2': float(base[0]),
#         'Y2': float(base[1])
#     })
    
#     # Subsequent rows: short axes
#     for axis_coords in short_axes:
#         data.append({
#             'X1': float(axis_coords[0]),
#             'Y1': float(axis_coords[1]),
#             'X2': float(axis_coords[2]),
#             'Y2': float(axis_coords[3])
#         })
    
#     return pd.DataFrame(data)

############ End traces from mask estimation