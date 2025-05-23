import cv2
import numpy as np
import logging

from typing import List, Tuple, Optional
from .embedding import EmbeddingExtractor

# We'll create our own simplified version of StrongSORT components
# to avoid import errors from the original codebase

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space includes the bounding box center position (x, y),
    aspect ratio a, height h, and their respective velocities.
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.
        
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Compute Kalman gain - avoid using cho_factor
        # Use regular matrix operations instead
        try:
            # Use numpy's built-in solve function for the system of linear equations
            tmp = np.linalg.inv(projected_cov)
            kalman_gain = np.dot(covariance, np.dot(self._update_mat.T, tmp))
        except np.linalg.LinAlgError:
            # If matrix is singular, use a more robust approach with regularization
            identity = np.eye(projected_cov.shape[0]) * 1e-8
            tmp = np.linalg.inv(projected_cov + identity)
            kalman_gain = np.dot(covariance, np.dot(self._update_mat.T, tmp))
            
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        # Use simplified Mahalanobis distance calculation
        distances = []
        try:
            # Try to use matrix inverse (more accurate but may fail)
            inv_covariance = np.linalg.inv(covariance)
            for measurement in measurements:
                d = measurement - mean
                distances.append(np.dot(d.T, np.dot(inv_covariance, d)))
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if matrix inversion fails
            for measurement in measurements:
                d = measurement - mean
                distances.append(np.sum(d * d))
                
        return np.array(distances)

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        """Initialize a track.
        """
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.n_init = n_init
        self.max_age = max_age
        self.state = TrackState.Tentative
        self.features = []
        self.is_violator = False  # Flag to indicate if this track represents a vehicle that violated rules
        if feature is not None:
            self.features.append(feature)
            
        # Add trajectory history
        self.trajectory = [(int(mean[0]), int(mean[1]))]  # Store (x,y) center positions
        self.bbox_history = [self.to_tlwh()]  # Store full bounding boxes
        self.max_trajectory_len = 30  # Maximum number of positions to store

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        
        # Update trajectory with predicted position
        center_x = int(self.mean[0])
        center_y = int(self.mean[1])
        self.trajectory.append((center_x, center_y))
        self.bbox_history.append(self.to_tlwh())
        
        # Limit trajectory length
        if len(self.trajectory) > self.max_trajectory_len:
            self.trajectory = self.trajectory[-self.max_trajectory_len:]
            self.bbox_history = self.bbox_history[-self.max_trajectory_len:]

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        
        # Update trajectory
        center_x = int(self.mean[0])
        center_y = int(self.mean[1])
        self.trajectory.append((center_x, center_y))
        self.bbox_history.append(self.to_tlwh())
        
        # Limit trajectory length
        if len(self.trajectory) > self.max_trajectory_len:
            self.trajectory = self.trajectory[-self.max_trajectory_len:]
            self.bbox_history = self.bbox_history[-self.max_trajectory_len:]

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age and not self.is_violator:
            # Only delete tracks if they are not violators and have exceeded max_age
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

class Detection:
    """
    This class represents a bounding box detection in a single image.
    """

    def __init__(self, tlwh, confidence=1.0, feature=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32) if feature is not None else None

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

class NearestNeighborDistanceMetric:
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.
    """

    def __init__(self, metric, matching_threshold, budget=None):
        self.metric = metric
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def distance(self, features, targets):
        """Compute distance between features and targets.
        """
        cost_matrix = np.zeros((len(features), len(targets)))
        for i, feature in enumerate(features):
            cost_matrix[i, :] = self._metric(feature, targets)
        return cost_matrix

    def _metric(self, feature, targets):
        # Cosine distance
        dist = 1. - np.dot(targets, feature.T) / (
            np.linalg.norm(targets, axis=1) * np.linalg.norm(feature))
        return dist

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

class Tracker:
    """
    This is the multi-target tracker using Kalman filter and Hungarian algorithm
    """
    def __init__(self, metric, max_age=30, n_init=3):
        self.metric = metric
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        
        # Adjusted parameters for matching
        self.max_iou_distance = 0.9    # Increased from 0.7 to be more lenient with IoU
        self.max_cosine_distance = 0.6  # Increased from 0.3 to allow more appearance variation
        self.appearance_weight = 0.4    # Reduced from 0.7 to rely more on spatial information
        self.min_confidence = 0.3      # Minimum detection confidence to consider
        
    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        # Handle empty detections
        if not detections:
            # Mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()
            # Remove deleted tracks
            self.tracks = [t for t in self.tracks if not t.is_deleted()]
            return

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
            
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _match(self, detections):
        if not detections:
            return [], list(range(len(self.tracks))), []
            
        if not self.tracks:
            return [], [], list(range(len(detections)))
            
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
            
        # Associate confirmed tracks using appearance features and IoU
        matches_a, unmatched_tracks_a, unmatched_detections = \
            self._matching_cascade(detections, confirmed_tracks)
            
        # Associate remaining tracks using IoU only
        iou_track_candidates = unconfirmed_tracks + unmatched_tracks_a
        matches_b, unmatched_tracks_b, unmatched_detections = \
            self._match_iou(detections, iou_track_candidates, unmatched_detections)
            
        matches = matches_a + matches_b
        unmatched_tracks = unmatched_tracks_b
            
        return matches, unmatched_tracks, unmatched_detections

    def _matching_cascade(self, detections, track_indices):
        if len(track_indices) == 0 or len(detections) == 0:
            return [], track_indices, list(range(len(detections)))
            
        # Compute combined cost matrix
        cost_matrix = np.zeros((len(track_indices), len(detections)))
        
        for row, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            
            # Get track box and detection boxes
            track_box = track.to_tlwh()
            det_boxes = np.array([d.tlwh for d in detections])
            
            # Calculate IoU distances
            iou_dists = np.array([1 - iou(track_box, det_box) for det_box in det_boxes])
            
            # Calculate appearance (cosine) distances if features available
            has_features = (track.features and len(track.features) > 0 and 
                          len(detections) > 0 and detections[0].feature is not None)
                          
            if has_features:
                # Use average of last N features for more stable matching
                n_features = min(5, len(track.features))
                track_features = np.array(track.features[-n_features:])
                track_feature = np.mean(track_features, axis=0)
                
                det_features = np.array([d.feature for d in detections])
                feature_dists = np.zeros(len(detections))
                
                for i, det_feature in enumerate(det_features):
                    if det_feature is None:
                        feature_dists[i] = 1.0  # Maximum distance for missing features
                        continue
                    cosine_dist = 1 - np.dot(det_feature, track_feature) / (
                        np.linalg.norm(det_feature) * np.linalg.norm(track_feature))
                    feature_dists[i] = cosine_dist
                
                # Add motion prediction component using Kalman filter
                predicted_pos = track.mean[:2]  # Get predicted x,y position
                det_centers = np.array([[d.tlwh[0] + d.tlwh[2]/2, d.tlwh[1] + d.tlwh[3]/2] for d in detections])
                motion_dists = np.linalg.norm(det_centers - predicted_pos, axis=1)
                motion_dists = motion_dists / (np.max(motion_dists) + 1e-7)  # Normalize to [0,1]
                
                # Combine all distance metrics with weights
                cost_matrix[row] = (
                    self.appearance_weight * feature_dists +
                    (1 - self.appearance_weight) * (0.7 * iou_dists + 0.3 * motion_dists)
                )
            else:
                # If no appearance features, use IoU and motion only
                predicted_pos = track.mean[:2]
                det_centers = np.array([[d.tlwh[0] + d.tlwh[2]/2, d.tlwh[1] + d.tlwh[3]/2] for d in detections])
                motion_dists = np.linalg.norm(det_centers - predicted_pos, axis=1)
                motion_dists = motion_dists / (np.max(motion_dists) + 1e-7)
                
                cost_matrix[row] = 0.7 * iou_dists + 0.3 * motion_dists

        # Apply confidence gating - use large value instead of inf
        large_cost = 1e5
        for col, detection in enumerate(detections):
            if detection.confidence < self.min_confidence:
                cost_matrix[:, col] = large_cost
                
        # Apply distance gating
        max_dist = self.max_cosine_distance
        cost_matrix[cost_matrix > max_dist] = large_cost
        
        # Ensure cost matrix is finite
        cost_matrix = np.nan_to_num(cost_matrix, nan=large_cost, posinf=large_cost, neginf=large_cost)
        
        try:
            # Run Hungarian algorithm
            indices = linear_assignment(cost_matrix)
            
            matches, unmatched_tracks, unmatched_detections = [], [], []
            for col, detection_idx in enumerate(range(len(detections))):
                if col not in indices[:, 1]:
                    unmatched_detections.append(detection_idx)
            for row, track_idx in enumerate(track_indices):
                if row not in indices[:, 0]:
                    unmatched_tracks.append(track_idx)
            for row, col in indices:
                track_idx = track_indices[row]
                detection_idx = col
                if cost_matrix[row, col] >= large_cost * 0.9:  # Use threshold slightly below large_cost
                    unmatched_tracks.append(track_idx)
                    unmatched_detections.append(detection_idx)
                else:
                    matches.append((track_idx, detection_idx))
                    
        except Exception as e:
            # Fallback to greedy matching if Hungarian algorithm fails
            matches, unmatched_tracks, unmatched_detections = [], [], []
            used_detections = set()
            
            # Sort tracks by age (older tracks first)
            track_indices_with_age = [(idx, self.tracks[idx].age) for idx in track_indices]
            track_indices_with_age.sort(key=lambda x: x[1], reverse=True)
            
            for track_idx, _ in track_indices_with_age:
                best_match = None
                min_cost = large_cost
                
                for det_idx in range(len(detections)):
                    if det_idx in used_detections:
                        continue
                        
                    cost = cost_matrix[track_indices.index(track_idx), det_idx]
                    if cost < min_cost:
                        min_cost = cost
                        best_match = det_idx
                
                if best_match is not None and min_cost < large_cost * 0.9:
                    matches.append((track_idx, best_match))
                    used_detections.add(best_match)
                else:
                    unmatched_tracks.append(track_idx)
            
            unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
                    
        return matches, unmatched_tracks, unmatched_detections

    def _match_iou(self, detections, track_indices, detection_indices):
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], track_indices, detection_indices
            
        # Calculate IoU cost matrix
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        large_cost = 1e5
        
        for row, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            track_box = track.to_tlwh()
            
            # Get predicted position from Kalman filter
            predicted_pos = track.mean[:2]
            
            for col, det_idx in enumerate(detection_indices):
                det_box = detections[det_idx].tlwh
                
                # Calculate IoU
                iou_dist = 1 - iou(track_box, det_box)
                
                # Calculate motion distance
                det_center = np.array([det_box[0] + det_box[2]/2, det_box[1] + det_box[3]/2])
                motion_dist = np.linalg.norm(det_center - predicted_pos)
                motion_dist = min(1.0, motion_dist / 100.0)  # Normalize with max expected motion
                
                # Combine IoU and motion with weights
                cost_matrix[row, col] = 0.7 * iou_dist + 0.3 * motion_dist
                
        # Apply gating
        cost_matrix[cost_matrix > self.max_iou_distance] = large_cost
        
        # Ensure cost matrix is finite
        cost_matrix = np.nan_to_num(cost_matrix, nan=large_cost, posinf=large_cost, neginf=large_cost)
        
        try:
            # Run Hungarian algorithm
            indices = linear_assignment(cost_matrix)
            
            matches, unmatched_tracks, unmatched_detections = [], [], []
            for col, detection_idx in enumerate(detection_indices):
                if col not in indices[:, 1]:
                    unmatched_detections.append(detection_idx)
            for row, track_idx in enumerate(track_indices):
                if row not in indices[:, 0]:
                    unmatched_tracks.append(track_idx)
            for row, col in indices:
                track_idx = track_indices[row]
                detection_idx = detection_indices[col]
                if cost_matrix[row, col] >= large_cost * 0.9:
                    unmatched_tracks.append(track_idx)
                    unmatched_detections.append(detection_idx)
                else:
                    matches.append((track_idx, detection_idx))
                    
        except Exception as e:
            # Fallback to greedy matching if Hungarian algorithm fails
            matches, unmatched_tracks, unmatched_detections = [], [], []
            used_detections = set()
            
            # Sort tracks by age
            track_indices_with_age = [(idx, self.tracks[idx].age) for idx in track_indices]
            track_indices_with_age.sort(key=lambda x: x[1], reverse=True)
            
            for track_idx, _ in track_indices_with_age:
                best_match = None
                min_cost = large_cost
                
                for det_idx in detection_indices:
                    if det_idx in used_detections:
                        continue
                        
                    cost = cost_matrix[track_indices.index(track_idx), det_idx]
                    if cost < min_cost:
                        min_cost = cost
                        best_match = det_idx
                
                if best_match is not None and min_cost < large_cost * 0.9:
                    matches.append((track_idx, best_match))
                    used_detections.add(best_match)
                else:
                    unmatched_tracks.append(track_idx)
            
            unmatched_detections = [i for i in detection_indices if i not in used_detections]
                    
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1

class VehicleTracker:
    """Vehicle tracking class using simplified StrongSORT"""

    def __init__(self, max_cosine_distance: float = 0.4, nn_budget: int = 100):
        """
        Initialize the StrongSORT vehicle tracker

        Args:
            max_cosine_distance: Threshold for feature distance metric
            nn_budget: Maximum size of the appearance descriptor gallery
        """
        self.logger = logging.getLogger("VehicleTracker")

        # Initialize the tracker with increased max_age to keep vehicles tracked longer
        self.metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(self.metric, max_age=60)  # Increase max_age from default 30

        # Track IDs and boxes
        self.next_id = 1
        self.tracked_boxes = []
        self.embedder = EmbeddingExtractor()
        self.violation_ids = set()  # Set to store IDs of vehicles that violated rules

        self.logger.info(f"Initialized VehicleTracker with simplified StrongSORT")

    def _create_detections(self, frame, boxes: List[Tuple[int, int, int, int]],
                           confidences: Optional[List[float]] = None):
        """
        Create Detection objects from bounding boxes

        Args:
            boxes: List of (x, y, w, h) bounding boxes
            confidences: List of confidence scores (optional)

        Returns:
            List of Detection objects
        """
        if confidences is None:
            confidences = [1.0] * len(boxes)

        detection_list = []
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            roi = frame[y:y2, x:x2]

            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                continue

            feature = self.embedder.extract(roi)
            detection_list.append(Detection((x, y, w, h), conf, feature))

        return detection_list

    def add_objects(self, frame, boxes: List[Tuple[int, int, int, int]]):
        """
        Add objects to track

        Args:
            frame: Current frame
            boxes: List of (x, y, w, h) bounding boxes to track

        Returns:
            List of assigned tracker IDs
        """
        filtered_boxes = []

        # Filter invalid boxes
        for box in boxes:
            x, y, w, h = box

            # Ensure box is within frame bounds
            if (x < 0 or y < 0 or w <= 0 or h <= 0 or
                x + w > frame.shape[1] or y + h > frame.shape[0]):
                continue

            filtered_boxes.append(box)

        # Create Detection objects and update tracker
        detections = self._create_detections(frame, filtered_boxes)

        # Predict then update
        self.tracker.predict()
        self.tracker.update(detections)

        # Get assigned IDs from tracks
        assigned_ids = []
        self.tracked_boxes = []

        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                # Get track box in (x, y, w, h) format
                bbox = track.to_tlwh()
                x, y, w, h = [int(v) for v in bbox]

                self.tracked_boxes.append((x, y, w, h))
                assigned_ids.append(track.track_id)

        return assigned_ids

    def add_violation_id(self, track_id):
        """Add a track ID to the set of violation IDs"""
        self.violation_ids.add(track_id)
        
    def is_violation_id(self, track_id):
        """Check if a track ID is in the set of violation IDs"""
        return track_id in self.violation_ids
        
    def update(self, frame, detections, stop_line_y=None):
        """
        Update trackers with new frame and detections

        Args:
            frame: Current frame
            detections: List of detections in format [x1, y1, x2, y2, conf, class_id]
            stop_line_y: Optional y-coordinate of the stop line. If provided, vehicles
                         above this line will be ignored for new detections, but tracked
                         vehicles will continue to be tracked.

        Returns:
            Tuple of (boxes, ids) where:
                boxes: List of current (x, y, w, h) boxes
                ids: List of corresponding tracker IDs
        """
        # Filter out new detections above the stop line if stop_line_y is provided
        filtered_detections = []
        if stop_line_y is not None:
            for det in detections:
                x1, y1, x2, y2, conf, class_id = det
                # Calculate vehicle's bottom y-coordinate (use this as reference point)
                bottom_y = y2
                # Only keep vehicles that are below or crossing the stop line for new detections
                if bottom_y >= stop_line_y:
                    filtered_detections.append(det)
        else:
            filtered_detections = detections

        # Process detections to the right format
        processed_boxes = []
        confidences = []

        for det in filtered_detections:
            x1, y1, x2, y2, conf, _ = det
            w = x2 - x1
            h = y2 - y1
            # Convert to (x, y, w, h) format
            processed_boxes.append((x1, y1, w, h))
            confidences.append(conf)

        # Create Detection objects and update tracker
        detection_objects = self._create_detections(frame, processed_boxes, confidences)

        # Update tracker's track management to prevent violator tracks from being deleted
        for track in self.tracker.tracks:
            if track.track_id in self.violation_ids:
                track.is_violator = True
            else:
                track.is_violator = False

        # Predict then update
        self.tracker.predict()
        self.tracker.update(detection_objects)

        # Extract current tracks without filtering by the stop line
        result_boxes = []
        result_ids = []
        self.tracked_boxes = []

        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                # Get track box in (x, y, w, h) format
                bbox = track.to_tlwh()
                x, y, w, h = [int(v) for v in bbox]

                # Skip invalid boxes or out-of-bounds
                if (w <= 0 or h <= 0 or x < 0 or y < 0 or
                    x + w > frame.shape[1] or y + h > frame.shape[0]):
                    continue

                # Keep tracking all vehicles, regardless of stop line position
                result_boxes.append((x, y, w, h))
                result_ids.append(track.track_id)
                self.tracked_boxes.append((x, y, w, h))

        return result_boxes, result_ids

    def remove_tracker(self, track_id: int):
        """
        Remove a tracker with the specified ID

        Args:
            track_id: ID of tracker to remove
        """
        for i, track in enumerate(self.tracker.tracks):
            if track.track_id == track_id:
                self.tracker.tracks.pop(i)
                return

    def clear(self):
        """Clear all trackers"""
        self.tracker = Tracker(self.metric)
        self.tracked_boxes = []

def iou(box1, box2):
    """Calculate intersection over union between two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Calculate areas
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return max(0.0, min(iou, 1.0))

def linear_assignment(cost_matrix):
    """
    Solve the linear assignment problem using the Hungarian algorithm
    """
    try:
        import scipy.optimize
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        return np.array(list(zip(row_ind, col_ind)))
    except ImportError:
        # Fallback to greedy matching if scipy is not available
        matches = []
        if cost_matrix.size == 0:
            return np.array([])
            
        # Greedy matching
        cost_matrix = cost_matrix.copy()
        rows, cols = cost_matrix.shape
        used_rows = set()
        used_cols = set()
        
        while True:
            if len(used_rows) == rows or len(used_cols) == cols:
                break
                
            # Find minimum cost
            min_val = float('inf')
            min_row = -1
            min_col = -1
            
            for i in range(rows):
                if i in used_rows:
                    continue
                for j in range(cols):
                    if j in used_cols:
                        continue
                    if cost_matrix[i,j] < min_val:
                        min_val = cost_matrix[i,j]
                        min_row = i
                        min_col = j
            
            if min_row == -1 or min_col == -1:
                break
                
            matches.append([min_row, min_col])
            used_rows.add(min_row)
            used_cols.add(min_col)
            
        return np.array(matches)