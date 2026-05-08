"""
Classical Event Noise Filtering Algorithms

This module implements various classical (non-learning-based) event noise filters
with hardware operation counting for complexity analysis.

Each filter returns:
    - predictions: Binary labels (0=signal, 1=noise) for each event
    - hw_ops: Dictionary of hardware operation counts
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict
from config import cfg


class BaseEventFilter(ABC):
    """Abstract base class for event filters."""
    
    def __init__(self, name: str):
        self.name = name
        self.hw_ops = {
            'comparison': 0,
            'addition': 0,
            'multiplication': 0,
            'division': 0,
            'sqrt': 0,
            'exp': 0,
            'memory_access': 0,
        }
    
    @abstractmethod
    def filter_events(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Filter events and return predictions with hardware operation count.
        
        Args:
            events: (N, 5) array [label, x, y, t, polarity]
                    Note: label is ground truth, not used by filter
        
        Returns:
            predictions: (N,) array of binary predictions (0=signal, 1=noise)
            hw_ops: Dictionary of hardware operation counts
        """
        pass
    
    def reset_hw_ops(self):
        """Reset hardware operation counters."""
        for key in self.hw_ops:
            self.hw_ops[key] = 0
    
    def get_total_ops(self) -> int:
        """Calculate total weighted hardware operations."""
        total = 0
        for op_type, count in self.hw_ops.items():
            total += count * cfg.HW_OP_COSTS[op_type]
        return total


class BAF(BaseEventFilter):
    """
    Background Activity Filter (BAF)
    
    Based on: jAER BackgroundActivityFilter.java
    
    Principle: Judges as signal if there is at least one event within the time window in the 3x3 neighborhood.
    Checks only 8 neighbors, excluding the pixel itself (filterHotPixels mode).
    This is effectively the same as STCF with k=1.
    """
    
    def __init__(self, time_window: float = 0.01):
        super().__init__("BAF")
        self.time_window = time_window
    
    def filter_events(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        self.reset_hw_ops()
        
        if len(events) == 0:
            return np.array([]), self.hw_ops.copy()
        
        x_coords = events[:, 1].astype(int)
        y_coords = events[:, 2].astype(int)
        timestamps = events[:, 3]
        
        N = len(events)
        predictions = np.ones(N, dtype=np.uint8)  # Initialize as noise (1)
        
        # Track last event time for each pixel (2D timestamp image)
        last_event_time = {}
        
        for i in range(N):
            x, y, t = x_coords[i], y_coords[i], timestamps[i]
            pixel_key = (x, y)
            
            # Check 3x3 neighborhood (excluding self, like Java's filterHotPixels)
            ncorrelated = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue  # Exclude self (filterHotPixels)
                    
                    neighbor_key = (x + dx, y + dy)
                    self.hw_ops['addition'] += 2
                    self.hw_ops['memory_access'] += 1
                    
                    if neighbor_key in last_event_time:
                        time_diff = t - last_event_time[neighbor_key]
                        self.hw_ops['addition'] += 1
                        self.hw_ops['comparison'] += 1
                        
                        if time_diff <= self.time_window:
                            ncorrelated += 1
                            predictions[i] = 0  # Signal
                            break  # Only need to find one
                
                if predictions[i] == 0:
                    break
            
            # Update last event time for this pixel
            last_event_time[pixel_key] = t
            self.hw_ops['memory_access'] += 1
        
        return predictions, self.hw_ops.copy()


class BAF_SinglePixel(BaseEventFilter):
    """
    Single Pixel Background Activity Filter
    
    Principle: Checks previous events only at the same pixel (no neighbors checked)
    Stricter filtering - requires consecutive events at the same pixel to pass
    
    This is the original simplest BAF definition:
    "Signal if there was a previous event at the same location within time"
    """
    
    def __init__(self, time_window: float = 0.01):
        super().__init__("BAF_SinglePixel")
        self.time_window = time_window
    
    def filter_events(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        self.reset_hw_ops()
        
        if len(events) == 0:
            return np.array([]), self.hw_ops.copy()
        
        x_coords = events[:, 1].astype(int)
        y_coords = events[:, 2].astype(int)
        timestamps = events[:, 3]
        
        N = len(events)
        predictions = np.ones(N, dtype=np.uint8)  # Initialize as noise (1)
        
        # Track last event time for each pixel
        last_event_time = {}
        
        for i in range(N):
            x, y, t = x_coords[i], y_coords[i], timestamps[i]
            pixel_key = (x, y)
            
            self.hw_ops['memory_access'] += 1
            
            # Check previous event only at the same pixel
            if pixel_key in last_event_time:
                time_diff = t - last_event_time[pixel_key]
                self.hw_ops['addition'] += 1
                self.hw_ops['comparison'] += 1
                
                if time_diff <= self.time_window:
                    predictions[i] = 0  # Signal
            
            # Update last event time
            last_event_time[pixel_key] = t
            self.hw_ops['memory_access'] += 1
        
        return predictions, self.hw_ops.copy()


class RefractoryFilter(BaseEventFilter):
    """
    Refractory Period Filter
    
    Principle: Remove events from the same pixel occurring too quickly.
    Mimics the biological refractory period of neurons.
    """
    
    def __init__(self, refractory_period: float = 0.001):
        super().__init__("Refractory")
        self.refractory_period = refractory_period
    
    def filter_events(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        self.reset_hw_ops()
        
        if len(events) == 0:
            return np.array([]), self.hw_ops.copy()
        
        x_coords = events[:, 1].astype(int)
        y_coords = events[:, 2].astype(int)
        timestamps = events[:, 3]
        
        N = len(events)
        predictions = np.zeros(N, dtype=np.uint8)  # Initialize as signal (0)
        
        last_event_time = {}
        
        for i in range(N):
            x, y, t = x_coords[i], y_coords[i], timestamps[i]
            pixel_key = (x, y)
            
            self.hw_ops['memory_access'] += 1
            
            if pixel_key in last_event_time:
                time_diff = t - last_event_time[pixel_key]
                self.hw_ops['addition'] += 1
                self.hw_ops['comparison'] += 1
                
                if time_diff < self.refractory_period:
                    predictions[i] = 1  # Noise (too fast)
                    continue
            
            last_event_time[pixel_key] = t
            self.hw_ops['memory_access'] += 1
        
        return predictions, self.hw_ops.copy()


class STCF(BaseEventFilter):
    """
    Spatio-Temporal Correlation Filter (STCF)
    
    Principle: Keep events that have spatial neighbors within a temporal window.
    An event is signal if it has sufficient correlated neighbors in space-time.
    """
    
    def __init__(self, spatial_radius: int = 1, temporal_window: float = 0.005, 
                 min_neighbors: int = 1):
        super().__init__("STCF")
        self.spatial_radius = spatial_radius
        self.temporal_window = temporal_window
        self.min_neighbors = min_neighbors
    
    def filter_events(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        self.reset_hw_ops()
        
        if len(events) == 0:
            return np.array([]), self.hw_ops.copy()
        
        x_coords = events[:, 1].astype(int)
        y_coords = events[:, 2].astype(int)
        timestamps = events[:, 3]
        
        N = len(events)
        predictions = np.ones(N, dtype=np.uint8)  # Initialize as noise
        
        # Build spatial-temporal index: dict of (x, y) -> list of (time, index, polarity)
        spatial_index = {}
        
        # Add Polarity: events[:, 4]
        polarities = events[:, 4].astype(int)
        
        for i in range(N):
            x, y, t, p = x_coords[i], y_coords[i], timestamps[i], polarities[i]
            pixel_key = (x, y)
            
            if pixel_key not in spatial_index:
                spatial_index[pixel_key] = []
            spatial_index[pixel_key].append((t, i, p))
            self.hw_ops['memory_access'] += 3  # read and write (t, i, p)
        
        # For each event, check neighbors
        for i in range(N):
            x, y, t, p = x_coords[i], y_coords[i], timestamps[i], polarities[i]
            neighbor_count = 0
            
            # Check spatial neighborhood
            for dx in range(-self.spatial_radius, self.spatial_radius + 1):
                for dy in range(-self.spatial_radius, self.spatial_radius + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip self
                    
                    neighbor_key = (x + dx, y + dy)
                    self.hw_ops['addition'] += 2  # x+dx, y+dy
                    self.hw_ops['memory_access'] += 1
                    
                    if neighbor_key in spatial_index:
                        # Check temporal correlation AND Polarity
                        for neighbor_t, neighbor_idx, neighbor_p in spatial_index[neighbor_key]:
                            time_diff = abs(t - neighbor_t)
                            self.hw_ops['addition'] += 1  # subtraction
                            self.hw_ops['comparison'] += 2 # time, polarity
                            
                            # Polarity Check added
                            if time_diff <= self.temporal_window and neighbor_idx != i and neighbor_p == p:
                                neighbor_count += 1
                                self.hw_ops['addition'] += 1
                                
                                if neighbor_count >= self.min_neighbors:
                                    predictions[i] = 0  # Signal
                                    break
                    
                    if predictions[i] == 0:
                        break
                
                if predictions[i] == 0:
                    break
        
        return predictions, self.hw_ops.copy()


class NNFilter(BaseEventFilter):
    """
    Nearest Neighbor (NN) Filter
    
    Principle: Requires at least N spatial neighbors within a time window.
    Similar to STCF but focuses on spatial correlation.
    """
    
    def __init__(self, spatial_radius: int = 1, temporal_window: float = 0.01,
                 min_neighbors: int = 2):
        super().__init__("NN")
        self.spatial_radius = spatial_radius
        self.temporal_window = temporal_window
        self.min_neighbors = min_neighbors
    
    def filter_events(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        self.reset_hw_ops()
        
        if len(events) == 0:
            return np.array([]), self.hw_ops.copy()
        
        x_coords = events[:, 1].astype(int)
        y_coords = events[:, 2].astype(int)
        timestamps = events[:, 3]
        
        N = len(events)
        predictions = np.ones(N, dtype=np.uint8)
        
        # Build temporal index for efficient neighbor search
        # Store events in time-sorted order with spatial info
        spatial_index = {}
        
        for i in range(N):
            x, y, t = x_coords[i], y_coords[i], timestamps[i]
            pixel_key = (x, y)
            
            if pixel_key not in spatial_index:
                spatial_index[pixel_key] = []
            spatial_index[pixel_key].append((t, i))
            self.hw_ops['memory_access'] += 2
        
        # Check each event
        for i in range(N):
            x, y, t = x_coords[i], y_coords[i], timestamps[i]
            neighbor_count = 0
            
            # Search in spatial neighborhood
            for dx in range(-self.spatial_radius, self.spatial_radius + 1):
                for dy in range(-self.spatial_radius, self.spatial_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor_key = (x + dx, y + dy)
                    self.hw_ops['addition'] += 2
                    self.hw_ops['memory_access'] += 1
                    
                    if neighbor_key in spatial_index:
                        for neighbor_t, neighbor_idx in spatial_index[neighbor_key]:
                            time_diff = abs(t - neighbor_t)
                            self.hw_ops['addition'] += 1
                            self.hw_ops['comparison'] += 1
                            
                            if time_diff <= self.temporal_window:
                                neighbor_count += 1
                                self.hw_ops['addition'] += 1
                                
                                if neighbor_count >= self.min_neighbors:
                                    predictions[i] = 0
                                    break
                    
                    if predictions[i] == 0:
                        break
                
                if predictions[i] == 0:
                    break
        
        return predictions, self.hw_ops.copy()


class BilateralFilter(BaseEventFilter):
    """
    Bilateral Filter for Events
    
    Principle: Weighted combination of spatial and temporal filtering.
    Uses Gaussian kernels for both spatial and temporal domains.
    """
    
    def __init__(self, spatial_sigma: float = 1.5, temporal_sigma: float = 0.005,
                 threshold: float = 2.0):
        super().__init__("Bilateral")
        self.spatial_sigma = spatial_sigma
        self.temporal_sigma = temporal_sigma
        self.threshold = threshold  # Now represents minimum number of correlated neighbors
        # Effective radius for spatial kernel (3-sigma rule)
        self.spatial_radius = int(np.ceil(3 * spatial_sigma))
    
    def filter_events(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        self.reset_hw_ops()
        
        if len(events) == 0:
            return np.array([]), self.hw_ops.copy()
        
        x_coords = events[:, 1].astype(int)
        y_coords = events[:, 2].astype(int)
        timestamps = events[:, 3]
        
        N = len(events)
        predictions = np.ones(N, dtype=np.uint8)
        
        # Build spatial index
        spatial_index = {}
        for i in range(N):
            x, y, t = x_coords[i], y_coords[i], timestamps[i]
            pixel_key = (x, y)
            
            if pixel_key not in spatial_index:
                spatial_index[pixel_key] = []
            spatial_index[pixel_key].append((t, i))
            self.hw_ops['memory_access'] += 2
        
        # Process each event
        for i in range(N):
            x, y, t = x_coords[i], y_coords[i], timestamps[i]
            neighbor_event_count = 0
            
            # Count weighted neighboring events
            for dx in range(-self.spatial_radius, self.spatial_radius + 1):
                for dy in range(-self.spatial_radius, self.spatial_radius + 1):
                    neighbor_key = (x + dx, y + dy)
                    self.hw_ops['addition'] += 2
                    self.hw_ops['memory_access'] += 1
                    
                    if neighbor_key in spatial_index:
                        # Spatial weight (Gaussian)
                        spatial_dist_sq = dx*dx + dy*dy
                        self.hw_ops['multiplication'] += 2
                        self.hw_ops['addition'] += 1
                        
                        spatial_weight = np.exp(-spatial_dist_sq / (2 * self.spatial_sigma**2))
                        self.hw_ops['multiplication'] += 2  # for denominator
                        self.hw_ops['division'] += 1
                        self.hw_ops['exp'] += 1
                        
                        for neighbor_t, neighbor_idx in spatial_index[neighbor_key]:
                            if neighbor_idx == i:
                                continue  # Skip self
                            
                            # Temporal weight (Gaussian)
                            time_diff = abs(t - neighbor_t)
                            self.hw_ops['addition'] += 1
                            
                            # Only count neighbors within temporal window
                            if time_diff <= 3 * self.temporal_sigma:  # 3-sigma rule
                                temporal_weight = np.exp(-time_diff**2 / (2 * self.temporal_sigma**2))
                                self.hw_ops['multiplication'] += 3  # time_diff^2, denominator
                                self.hw_ops['division'] += 1
                                self.hw_ops['exp'] += 1
                                
                                # Combined weight
                                weight = spatial_weight * temporal_weight
                                self.hw_ops['multiplication'] += 1
                                
                                # Count if weight is significant
                                if weight >= 0.1:  # Threshold for significant contribution
                                    neighbor_event_count += 1
                                    self.hw_ops['addition'] += 1
                                    self.hw_ops['comparison'] += 1
            
            # Decision: if enough correlated neighbors, mark as signal
            self.hw_ops['comparison'] += 1
            if neighbor_event_count >= self.threshold:
                predictions[i] = 0  # Signal
        
        return predictions, self.hw_ops.copy()

class STCF_Sub(BaseEventFilter):
    """
    Subsampling-based Spatiotemporal Correlation Filter (STCF_Sub)
    
    Based on: "Design of a Spatiotemporal Correlation Filter for Event-based Sensors"
    
    Principle: 
    - Subsample NxN pixel blocks into a single cell
    - Passes if there was a previous event within the time window dT in the same block
    - Can support other events in the same block including itself
    
    Memory: O(W/N * H/N) - One timestamp per block
    Ops/Event: ~4 (1 read, 1 comparison, 1 write, 1 addition)
    """
    
    def __init__(self, block_size: int = 2, time_window: float = 0.01):
        super().__init__("STCF_Sub")
        self.block_size = block_size  # 2x2, 4x4 등
        self.time_window = time_window  # dT
    
    def filter_events(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        self.reset_hw_ops()
        
        if len(events) == 0:
            return np.array([]), self.hw_ops.copy()
        
        x_coords = events[:, 1].astype(int)
        y_coords = events[:, 2].astype(int)
        timestamps = events[:, 3]
        
        N = len(events)
        predictions = np.ones(N, dtype=np.uint8)  # Initialize as noise
        
        # Save last event timestamp per block
        # Key: (block_x, block_y), Value: timestamp
        block_timestamps = {}
        
        for i in range(N):
            x, y, t = x_coords[i], y_coords[i], timestamps[i]
            
            # Calculate block coordinates (subsampling)
            block_x = x // self.block_size
            block_y = y // self.block_size
            block_key = (block_x, block_y)
            
            self.hw_ops['addition'] += 2  # Approximate with shift instead of division
            self.hw_ops['memory_access'] += 1  # read
            
            # Check if there is a previous event in the same block
            if block_key in block_timestamps:
                time_diff = t - block_timestamps[block_key]
                self.hw_ops['addition'] += 1  # subtraction
                self.hw_ops['comparison'] += 1
                
                if time_diff <= self.time_window:
                    predictions[i] = 0  # Signal (correlated)
            
            # 블록 타임스탬프 업데이트
            block_timestamps[block_key] = t
            self.hw_ops['memory_access'] += 1  # write
        
        return predictions, self.hw_ops.copy()


class ONF(BaseEventFilter):
    """
    O(N) Filter (ONF) - Order N Background Activity Filter
    
    Based on: jAER OrderNBackgroundActivityFilter.java
    Reference: Khodamoradi & Kastner 2018 IEEE Emerging Topics
    
    Principle: 
    - lastRowTs[y]: Last event timestamp of row y
    - lastColTs[x]: Last event timestamp of column x
    - lastXByRow[y]: Last event x-coordinate of row y
    - lastYByCol[x]: Last event y-coordinate of column x
    - Event pass condition: Event within time in adjacent row/col (±1) and adjacent coordinates (±1)
    
    Memory: O(W + H)
    """
    
    def __init__(self, time_window: float = 0.001, width: int = 1280, height: int = 720):
        super().__init__("ONF")
        self.time_window = time_window
        self.width = width
        self.height = height
    
    def filter_events(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        self.reset_hw_ops()
        
        if len(events) == 0:
            return np.array([]), self.hw_ops.copy()
        
        x_coords = events[:, 1].astype(int)
        y_coords = events[:, 2].astype(int)
        timestamps = events[:, 3]
        
        N = len(events)
        predictions = np.ones(N, dtype=np.uint8)  # Initialize as noise
        
        # Row/Column timestamp and coordinate arrays
        lastRowTs = np.full(self.height, -np.inf)
        lastColTs = np.full(self.width, -np.inf)
        lastXByRow = np.full(self.height, -1, dtype=int)  # x coordinate for each row
        lastYByCol = np.full(self.width, -1, dtype=int)   # y coordinate for each column
        
        for i in range(N):
            x, y, t = x_coords[i], y_coords[i], timestamps[i]
            
            # Clamp to valid range (edge events treated as noise per Java)
            if x <= 0 or y <= 0 or x >= self.width - 1 or y >= self.height - 1:
                # Save event and continue (edge events are noise)
                lastXByRow[y] = x
                lastYByCol[x] = y
                lastColTs[x] = t
                lastRowTs[y] = t
                continue
            
            passed = False
            
            # Check adjacent rows (y-1, y, y+1)
            for dy in range(-1, 2):
                row_idx = y + dy
                self.hw_ops['memory_access'] += 2
                self.hw_ops['addition'] += 2
                self.hw_ops['comparison'] += 2
                
                if lastRowTs[row_idx] != -np.inf:
                    time_diff = t - lastRowTs[row_idx]
                    x_diff = abs(lastXByRow[row_idx] - x)
                    
                    if time_diff < self.time_window and x_diff <= 1:
                        passed = True
                        break
            
            # Check adjacent columns (x-1, x, x+1)
            if not passed:
                for dx in range(-1, 2):
                    col_idx = x + dx
                    self.hw_ops['memory_access'] += 2
                    self.hw_ops['addition'] += 2
                    self.hw_ops['comparison'] += 2
                    
                    if lastColTs[col_idx] != -np.inf:
                        time_diff = t - lastColTs[col_idx]
                        y_diff = abs(lastYByCol[col_idx] - y)
                        
                        if time_diff < self.time_window and y_diff <= 1:
                            passed = True
                            break
            
            if passed:
                predictions[i] = 0  # Signal
            
            # Save event
            self.hw_ops['memory_access'] += 4
            lastXByRow[y] = x
            lastYByCol[x] = y
            lastColTs[x] = t
            lastRowTs[y] = t
        
        return predictions, self.hw_ops.copy()


def create_filter(filter_name: str) -> BaseEventFilter:
    """
    Factory function to create filter instances.
    
    Args:
        filter_name: Name of the filter ('BAF', 'STCF', 'Refractory', 'NN', 'Bilateral')
    
    Returns:
        Instantiated filter object
    """
    filter_configs = cfg.FILTER_CONFIGS
    
    if filter_name == 'BAF':
        return BAF(**filter_configs['BAF'])
    elif filter_name == 'STCF':
        return STCF(**filter_configs['STCF'])
    elif filter_name == 'Refractory':
        return RefractoryFilter(**filter_configs['Refractory'])
    elif filter_name == 'NN':
        return NNFilter(**filter_configs['NN'])
    elif filter_name == 'Bilateral':
        return BilateralFilter(**filter_configs['Bilateral'])
    elif filter_name == 'ONF':
        return ONF(**filter_configs.get('ONF', {'time_window': 0.01}))
    elif filter_name == 'STCF_Sub':
        return STCF_Sub(**filter_configs.get('STCF_Sub', {'block_size': 2, 'time_window': 0.01}))
    elif filter_name == 'BAF_SinglePixel':
        return BAF_SinglePixel(**filter_configs.get('BAF_SinglePixel', {'time_window': 0.01}))
    else:
        raise ValueError(f"Unknown filter: {filter_name}")


def get_all_filter_names():
    """Return list of all available filter names."""
    return ['BAF', 'BAF_SinglePixel', 'STCF', 'Refractory', 'NN', 'Bilateral', 'ONF', 'STCF_Sub']
