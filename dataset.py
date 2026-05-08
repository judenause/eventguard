# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
# from config import cfg # config_obj is passed as an argument instead of using cfg directly

class EventFrameLazyDataset(Dataset):
    """
    PyTorch Dataset for event frame data with lazy window loading.
    Generates windows on-the-fly in __getitem__ from pre-processed file data.
    Each item in data_list is expected to be a dictionary containing frame sequences
    for one original .npy file, as returned by data_processing.process_folder_to_frame_lists.
    """
    def __init__(self,
                 data_list: list[dict], # Return value from data_processing.process_folder_to_frame_lists
                 config_obj): # Instance of cfg from config.py
        """
        Args:
            data_list (list[dict]): A list of dictionaries. Each dictionary corresponds to one
                                    processed file and should contain keys like 'file_path',
                                    'input_frames', 'real_event_gt', 'noise_event_gt',
                                    'evaluation_mask'. The values for frame data are numpy arrays
                                    of shape (T, H, W).
            config_obj: The configuration object (e.g., cfg from config.py) containing
                        WINDOW_SIZE and WINDOW_STRIDE.
        """
        super().__init__()
        self.data_list = data_list
        self.window_size = config_obj.WINDOW_SIZE
        self.stride = config_obj.WINDOW_STRIDE
        self.use_augmentation = getattr(config_obj, 'USE_DATA_AUGMENTATION', False)

        self.index_map = [] # Maps a global window index to (file_index_in_data_list, window_start_frame_in_file)
        self.total_windows = 0

        if not self.data_list:
            print("Warning: EventFrameLazyDataset initialized with an empty data_list.")
            # total_windows remains 0, len(dataset) will be 0.
            return

        print(f"Initializing EventFrameLazyDataset with {len(self.data_list)} processed files...")
        # Pre-calculate the index map and total number of possible windows across all files
        for file_idx, file_entry_dict in enumerate(self.data_list):
            # Use 'num_frames' metadata (prevents full loading into RAM)
            num_frames_in_file = file_entry_dict['num_frames']

            # Calculate available windows for this file
            if num_frames_in_file >= self.window_size:
                num_windows_in_this_file = (num_frames_in_file - self.window_size) // self.stride + 1
                for i in range(num_windows_in_this_file):
                    # Save start frame index for each window
                    window_start_frame_idx = i * self.stride
                    self.index_map.append((file_idx, window_start_frame_idx))
                self.total_windows += num_windows_in_this_file
            # else:
                # Files with fewer frames than WINDOW_SIZE may have already been filtered during data_processing.
                # print(f"  File {file_idx} ({os.path.basename(file_entry_dict.get('file_path','N/A'))}) has {num_frames_in_file} frames, less than window_size {self.window_size}. No windows generated.")

        if self.total_windows == 0 and len(self.data_list) > 0:
            print(f"Warning: EventFrameLazyDataset - No valid windows could be generated from the {len(self.data_list)} provided files "
                  f"with window_size={self.window_size} and stride={self.stride}.")
        elif self.total_windows > 0 :
             print(f"EventFrameLazyDataset initialized successfully. Total window samples: {self.total_windows} across {len(self.data_list)} files.")


    def __len__(self) -> int:
        """Returns the total number of window samples across all files."""
        return self.total_windows

    def __getitem__(self, global_window_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates one windowed sample of data on-the-fly.

        Args:
            global_window_idx (int): The global index of the window sample to fetch
                                     (from 0 to total_windows - 1).

        Returns:
            A tuple containing PyTorch Tensors:
                - inputs_tensor (torch.Tensor): Input window (T_win, C=1, H, W).
                - real_gt_tensor (torch.Tensor): Real event GT window (T_win, H, W).
                - noise_gt_tensor (torch.Tensor): Noise event GT window (T_win, H, W).
                - eval_mask_tensor (torch.Tensor): Evaluation mask window (T_win, H, W).
                - is_new_file_tensor (torch.Tensor): Flag (1.0 or 0.0) indicating start of a new file.
        """
        if not (0 <= global_window_idx < self.total_windows):
            raise IndexError(f"Index {global_window_idx} is out of bounds for EventFrameLazyDataset of size {self.total_windows}.")

        # Find which file and which window index this global_window_idx belongs to
        # Slicing current window using NumPy slicing
        if self.index_map is None:
             raise ValueError("Index map not initialized.")

        file_idx, window_start_frame_in_file = self.index_map[global_window_idx]
        file_entry_dict = self.data_list[file_idx]
        
        # --- Lazy Loading with Mmap ---
        processed_path = file_entry_dict['processed_path']
        
        # Open file in mmap mode (Read-only, no full load)
        # Stacked Frame Shape: [T, 4, H, W]
        # 0:Input, 1:RealGT, 2:NoiseGT, 3:EvalMask
        mmap_data = np.load(processed_path, mmap_mode='r')
        
        start_idx = window_start_frame_in_file
        end_idx = start_idx + self.window_size
        
        # Slice only the required window (Disk I/O happens here)
        window_data = mmap_data[start_idx:end_idx] # Shape: [Window, 4, H, W]
        
        # Separate channels (No copy yet, still view if possible, but slicing usually copies)
        input_window_np = window_data[:, 0, :, :].copy()     # (T_win, H, W)
        real_gt_window_np = window_data[:, 1, :, :].copy()   # (T_win, H, W)
        noise_gt_window_np = window_data[:, 2, :, :].copy()  # (T_win, H, W)
        eval_mask_window_np = window_data[:, 3, :, :].copy() # (T_win, H, W)

        # --- Data Augmentation (Random Flip) ---
        if hasattr(self, 'use_augmentation') and self.use_augmentation:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                input_window_np = np.flip(input_window_np, axis=2).copy()
                real_gt_window_np = np.flip(real_gt_window_np, axis=2).copy()
                noise_gt_window_np = np.flip(noise_gt_window_np, axis=2).copy()
                eval_mask_window_np = np.flip(eval_mask_window_np, axis=2).copy()
            
            # Random Vertical Flip
            if np.random.rand() > 0.5:
                input_window_np = np.flip(input_window_np, axis=1).copy()
                real_gt_window_np = np.flip(real_gt_window_np, axis=1).copy()
                noise_gt_window_np = np.flip(noise_gt_window_np, axis=1).copy()
                eval_mask_window_np = np.flip(eval_mask_window_np, axis=1).copy()

        # Add channel dimension (C=1): (T_win, H, W) -> (T_win, 1, H, W)
        inputs_window_np_ch = np.expand_dims(input_window_np, axis=1)

        # Convert NumPy arrays to PyTorch tensors
        inputs_tensor = torch.FloatTensor(inputs_window_np_ch)
        real_gt_tensor = torch.FloatTensor(real_gt_window_np)
        noise_gt_tensor = torch.FloatTensor(noise_gt_window_np)
        eval_mask_tensor = torch.FloatTensor(eval_mask_window_np)

        # Check if this is the start of a new file (sequence)
        is_new_file = (window_start_frame_in_file == 0)
        is_new_file_tensor = torch.tensor(1.0 if is_new_file else 0.0, dtype=torch.float32)

        return inputs_tensor, real_gt_tensor, noise_gt_tensor, eval_mask_tensor, is_new_file_tensor


# --- (Reference) EventFrameWindowDataset (Original) ---
# This method generates all windows in advance and keeps them in memory. 
# It may not be suitable for large datasets.
class EventFrameWindowDataset(Dataset):
    """
    PyTorch Dataset for pre-generated event frame window samples.
    Assumes all windowed data (input, GTs, mask) is passed as large numpy arrays.
    """
    def __init__(self,
                 all_input_windows_np: np.ndarray,   # (N_total_win, T_win, H, W)
                 all_real_gt_windows_np: np.ndarray, # (N_total_win, T_win, H, W)
                 all_noise_gt_windows_np: np.ndarray,# (N_total_win, T_win, H, W)
                 all_eval_mask_windows_np: np.ndarray): # (N_total_win, T_win, H, W)
        """
        Args:
            all_input_windows_np: All windowed input frames.
            all_real_gt_windows_np: All windowed real event GTs.
            all_noise_gt_windows_np: All windowed noise event GTs.
            all_eval_mask_windows_np: All windowed evaluation masks.
        """
        if not (all_input_windows_np.shape[0] == all_real_gt_windows_np.shape[0] == \
                all_noise_gt_windows_np.shape[0] == all_eval_mask_windows_np.shape[0]):
            raise ValueError("All windowed numpy arrays must have the same number of samples (N_total_win).")

        self.input_windows = all_input_windows_np
        self.real_gt_windows = all_real_gt_windows_np
        self.noise_gt_windows = all_noise_gt_windows_np
        self.eval_mask_windows = all_eval_mask_windows_np

        if self.input_windows.ndim == 4: # (N_win, T_win, H, W)
            self.num_samples = self.input_windows.shape[0]
            self.window_len = self.input_windows.shape[1]
            self.height = self.input_windows.shape[2]
            self.width = self.input_windows.shape[3]
        else: # Unexpected dimensions
             raise ValueError(f"Input windows should be 4D (N_win, T_win, H, W), got {self.input_windows.ndim}D")


        print(f"EventFrameWindowDataset initialized with {self.num_samples} window samples.")
        print(f"  - Sample shape (Input): (T_win={self.window_len}, H={self.height}, W={self.width})")


    def __len__(self) -> int:
        """Returns the total number of pre-generated window samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates one sample of pre-generated windowed data.
        """
        # Add channel dimension (C=1): (T_win, H, W) -> (T_win, 1, H, W)
        inputs_np_ch = np.expand_dims(self.input_windows[idx], axis=1)

        # Fetch GT and mask windows for the given index
        real_gt_np = self.real_gt_windows[idx]
        noise_gt_np = self.noise_gt_windows[idx]
        eval_mask_np = self.eval_mask_windows[idx]

        # Convert NumPy arrays to PyTorch tensors
        return (torch.FloatTensor(inputs_np_ch),
                torch.FloatTensor(real_gt_np),
                torch.FloatTensor(noise_gt_np),
                torch.FloatTensor(eval_mask_np))