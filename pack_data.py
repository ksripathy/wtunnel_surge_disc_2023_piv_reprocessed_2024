#%% Base libraries
"""
This script sets up the environment for analyzing and visualizing data from a wind tunnel experiment.
It includes the following functionalities:

1. Importing necessary libraries:
    - numpy: for numerical operations
    - matplotlib: for plotting
    - os: for interacting with the operating system
    - copy: for deep copying objects
    - inspect: for inspecting live objects
    - fnmatch: for filename pattern matching
    - matlab.engine: for interfacing with MATLAB
    - scipy.interpolate: for interpolation methods
    - numpy.ma: for masked arrays
    - scipy.signal: for signal processing
    - IPython.display: for displaying HTML content in Jupyter notebooks
    - contourpy: for generating contour plots
    - matplotlib.colors: for handling color normalization

2. Initiating a MATLAB engine instance for running MATLAB code from Python.

3. Configuring relative file locations for source code, plots, and data directories.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib ipympl
import os
from copy import deepcopy
import glob
import inspect
import fnmatch
import matlab.engine
#Initiate matlab engine
mlab_eng1 = matlab.engine.start_matlab()

from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from numpy.ma import masked_array
from scipy.interpolate import CubicSpline
from scipy import signal
import pandas as pd
import pickle

# Use the following line to display preformatted text in Jupyter notebooks when running with ipympl backend.
from IPython.display import display, HTML
display(HTML("<style>pre { white-space: pre !important; }</style>"))

from contourpy import contour_generator
from matplotlib.colors import BoundaryNorm

#Configuring relative file locations
home_dir = os.path.abspath("")
plot_dir = os.path.join(home_dir,"plots")
data_dir = os.path.join(home_dir,"raw_data")

#%% Miscellenaeous functions
def low_pass_filt(sig, sampleFreq, cutOffFreq):
    """
    Apply a low-pass Butterworth filter to a signal.
    Parameters:
    sig (array-like): The input signal to be filtered.
    sampleFreq (float): The sampling frequency of the input signal.
    cutOffFreq (float): The cutoff frequency for the low-pass filter.
    Returns:
    array-like: The filtered signal.
    """
    
    Wn = cutOffFreq/(0.5 * sampleFreq)
    b, a = signal.butter(2, Wn)
    
    return signal.filtfilt(b, a, sig)

class FormatContour:
    """
    A class to format contour data using interpolation.
    Attributes:
    -----------
    interp_func : RegularGridInterpolator
        An interpolator function for the given grid data.
    Methods:
    --------
    __init__(grid_x, grid_y, grid_values):
        Initializes the FormatContour with grid data and values.
    __call__(x, y):
        Interpolates the value at the given (x, y) coordinates and returns a formatted string.
    """
    
    def __init__(self, grid_x, grid_y, grid_values):
        self.interp_func = RegularGridInterpolator((grid_x[:, 0], grid_y[0, :]), grid_values)

    def __call__(self, x, y):
        z = np.take(self.interp_func((x, y)), 0)
        return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)

#%% piv data structure definition
class planar_piv_data:
    """A class representing planar PIV data.
    Attributes:
    - deflt_grid_x: numpy.ndarray
        The default grid x-coordinates.
    - deflt_grid_z: numpy.ndarray
        The default grid z-coordinates.
    - grid_x: numpy.ndarray
        The grid x-coordinates.
    - grid_z: numpy.ndarray
        The grid z-coordinates.
    - deflt_grid_v_x: numpy.ndarray
        The default grid x-velocity.
    - grid_v_x: numpy.ndarray
        The grid x-velocity.
    - grid_v_z: numpy.ndarray
        The grid z-velocity.
    - grid_v_mag: numpy.ndarray
        The grid velocity magnitude.
    - grid_omega_y: numpy.ndarray
        The grid y-vorticity.
    - grid_omega_mag: numpy.ndarray
        The grid vorticity magnitude.
    - grid_r_xx: numpy.ndarray
        The grid xx Reynolds stress.
    - grid_r_xz: numpy.ndarray
        The grid xz Reynolds stress.
    - grid_r_zz: numpy.ndarray
        The grid zz Reynolds stress.
    - grid_std_v_x: numpy.ndarray
        The grid x-velocity standard deviation.
    - grid_std_v_z: numpy.ndarray
        The grid z-velocity standard deviation.
    - grid_valid_cell: numpy.ndarray
        The grid valid cell mask.
    Methods:
    - __init__(frame_i, frame_j, frame_data)
        Initializes the planar_piv_data object.
    - corr_frame(x_corr, z_corr, mode="absolute")
        Adjusts the grid_x and grid_z coordinates based on the given correction values.
    - reorient_frame(disc_xc, disc_zc)
        Reorients the frame of the grid by subtracting the given disc_xc and disc_zc values.
    - gen_data_intpr()
        Generates interpolated data for the object.
    """
    
    def __init__(self, frame_i, frame_j, frame_data):
        """
        Initializes the grid and velocity data for the given frames.
        Parameters:
        frame_i (int): Number of rows in the grid.
        frame_j (int): Number of columns in the grid.
        frame_data (numpy.ndarray): Array containing the grid and velocity data. 
            The array is expected to have the following columns:
            - Column 0: x-coordinates of the grid points.
            - Column 1: z-coordinates of the grid points.
            - Column 2: x-components of the velocity at the grid points.
            - Column 3: z-components of the velocity at the grid points.
            - Column 4: Magnitude of the velocity at the grid points.
            - Column 5: y-components of the vorticity at the grid points.
            - Column 6: Magnitude of the vorticity at the grid points.
            - Column 7: xx-components of the Reynolds stress at the grid points.
            - Column 8: xz-components of the Reynolds stress at the grid points.
            - Column 9: zz-components of the Reynolds stress at the grid points.
            - Column 10: Standard deviation of the x-components of the velocity.
            - Column 11: Standard deviation of the z-components of the velocity.
            - Column 12: Validity of the grid cells (boolean).
        """
        
        self.deflt_grid_x = frame_data[:,0].reshape(frame_i, frame_j, order='F')
        self.deflt_grid_z = frame_data[:,1].reshape(frame_i, frame_j, order='F')
        self.grid_x  = deepcopy(self.deflt_grid_x)
        self.grid_z = deepcopy(self.deflt_grid_z)
        self.deflt_grid_v_x = frame_data[:,2].reshape(frame_i, frame_j, order='F')
        self.grid_v_x = deepcopy(self.deflt_grid_v_x)
        self.grid_v_z = frame_data[:,3].reshape(frame_i, frame_j, order='F')
        self.grid_v_mag = frame_data[:,4].reshape(frame_i, frame_j, order='F')
        self.grid_omega_y = frame_data[:,5].reshape(frame_i, frame_j, order='F')
        self.grid_omega_mag = frame_data[:,6].reshape(frame_i, frame_j, order='F')
        self.grid_r_xx = frame_data[:,7].reshape(frame_i, frame_j, order='F')
        self.grid_r_xz = frame_data[:,8].reshape(frame_i, frame_j, order='F')
        self.grid_r_zz = frame_data[:,9].reshape(frame_i, frame_j, order='F')
        self.grid_std_v_x = frame_data[:,10].reshape(frame_i, frame_j, order='F')
        self.grid_std_v_z = frame_data[:,11].reshape(frame_i, frame_j, order='F')
        self.grid_valid_cell = frame_data[:,12].reshape(frame_i, frame_j, order='F').astype(bool)
        
    def corr_frame(self, x_corr, z_corr, mode = "absolute"):
        """
        Adjusts the grid_x and grid_z coordinates based on the given x_corr and z_corr values.
        Parameters:
        - x_corr: float or int
            The correction value to be added to the grid_x coordinates.
        - z_corr: float or int
            The correction value to be added to the grid_z coordinates.
        - mode: str, optional (default: "absolute")
            The mode of correction. It can be either "absolute" or "relative".
            - "absolute": Adds the correction values directly to the resetted grid_x and grid_z coordinates.
            - "relative": Adds the correction values to the existing grid_x and grid_z coordinates.
        Returns:
        None
        """
            
        if mode == "absolute":
            
            self.grid_x = deepcopy(self.deflt_grid_x)
            self.grid_z = deepcopy(self.deflt_grid_z)
            
            self.grid_x += x_corr
            self.grid_z += z_corr
            
        elif mode == "relative":
            
            self.grid_x += x_corr
            self.grid_z += z_corr
            
    def reorient_frame(self, disc_xc, disc_zc):
        """
        Davis PIV data has the flow direction from right to left.
        Reorients the frame of reference for the grid data by switching the flow direction, adjusting the 
        coordinates and gradients based on the provided disc center coordinates.
        Parameters:
        disc_xc (float): The x-coordinate of the disc center.
        disc_zc (float): The z-coordinate of the disc center.
        This method performs the following operations:
        - Resets the grid coordinates and velocity components to their default values.
        - Adjusts the grid coordinates by subtracting the disc center coordinates.
        - Inverts the x-coordinates and x-velocity components to switch the flow direction.
        - Inverts the y-component of the vorticity and the xz-component of the Reynolds stress to reflect the change in flow direction.
        - Computes the gradients of the velocity components with respect to the grid coordinates.
        - Scales down the grid coordinates by a factor of 200. 200 mm is the porous disc diameter. Use appropriate scaling for your use case.
        """
        
        
        self.grid_x = deepcopy(self.deflt_grid_x)
        self.grid_z = deepcopy(self.deflt_grid_z)
        self.grid_v_x = deepcopy(self.deflt_grid_v_x)
               
        self.grid_x -= disc_xc
        self.grid_z -= disc_zc
        
        self.grid_x *= -1
        self.grid_v_x *= -1
        self.grid_omega_y *= -1
        self.grid_r_xz *= -1
        self.grid_v_x_grad_x = np.gradient(self.grid_v_x, self.grid_x[:,0]*1e-3, axis=0)
        self.grid_v_x_grad_z = np.gradient(self.grid_v_x, self.grid_z[0,:]*1e-3, axis=1)
        self.grid_v_z_grad_x = np.gradient(self.grid_v_z, self.grid_x[:,0]*1e-3, axis=0)
        self.grid_v_z_grad_z = np.gradient(self.grid_v_z, self.grid_z[0,:]*1e-3, axis=1)
        self.grid_x /= 200
        self.grid_z /= 200
        
    def gen_data_intpr(self):
        """
        Generate interpolated data for the given object.
        This method checks if the interpolated data already exists for the object. If it does, the interpolation is skipped.
        Otherwise, it interpolates the dependent data using the NearestNDInterpolator.
        Parameters:
        - self: The object instance.
        Returns:
        - None
        """
                
        for name, value in inspect.getmembers(self):
            
            if fnmatch.fnmatch(name, "grid*intpr"): 
                return
        
        for name, value in inspect.getmembers(self):
            
            if fnmatch.fnmatch(name, "grid*") and name != "grid_x" and name != "grid_z" and name != "grid_valid_cell":
            
                setattr(self, name + "_intpr", NearestNDInterpolator((self.grid_x[self.grid_valid_cell], self.grid_z[self.grid_valid_cell]), getattr(self, name)[self.grid_valid_cell]))
                
class stitched_planar_piv_data:
    """
    Class for processing and analyzing stitched planar PIV (Particle Image Velocimetry) data.
    Attributes:
    phase_uid : str
        Unique identifier for the phase.
    piv_dt : float
        Time interval between PIV frames in milliseconds. Just for bookkeeping.
    v_surge : float
        Surge motion velocity.
    v_star_surge : float
        Non-dimensional surge velocity.
    Methods:
    __init__(phase_uid, piv_dt, v_surge, p_inf, rho_inf, v_inf):
        Initializes the stitched_planar_piv_data object with the given parameters.
    stitch_frames(frame0, frame1, x_trim=0.3, z_trim=0.025, frame_res=1.334/200, exp_factor=12):
        Stitches two frames together using overlapping weights.
    gen_data_intpr():
        Generates interpolated data for the given object.
    set_disc_mask(disc_mask_x_ustream=0.1, disc_mask_x_dstream=0.3, disc_mask_z_symm=0.7):
        Creates a disc mask to handle poor quality PIV data in the disc region.
    gen_unshadow_field(disc_mask_x_ustream=0.1, disc_mask_x_dstream=0.3, disc_mask_z_symm=0.7, disc_dia=200e-3, mlab_eng_obj=mlab_eng1):
        Generates an unshadowed field by interpolating and smoothing data in the disc region. This is the default method for generating unshadowed fields.
    get_unshadow_field(field_id, field_mask, x_ustr_indx, x_dstr_indx, mlab_eng_obj=mlab_eng1):
        Retrieves the unshadowed field using the outer boundaries of the field mask for interpolation. This allows the user to choose an arbitrary mask size.
    gen_pressure(p_inf, v_inf, rho_inf, mu=1.8e-5, disc_dia=200e-3, mlab_eng_obj=mlab_eng1):
        Reconstructs the pressure field from velocity and Reynolds stress data.
    """
    
    
    def __init__(self, phase_uid, piv_dt, v_surge, p_inf, rho_inf, v_inf):
        """Initializes the stitched_planar_piv_data object with the given parameters.

        Args:
            phase_uid (str): Unique identifier for the phase.
            piv_dt (float): Time interval between PIV frames in milliseconds. Just for bookkeeping.
            v_surge (float): Surge motion velocity.
            p_inf (float): Free stream pressure. Obtained from tunnel data.
            rho_inf (float): Free stream density. Obtained from tunnel data.
            v_inf (float): Free stream velocity. Obtained from tunnel data corrected using freestream piv flowfields.
        """
        
        self.phase_uid = phase_uid
        self.piv_dt = piv_dt
        self.v_surge = v_surge
        self.p_inf = p_inf
        self.rho_inf = rho_inf
        self.v_inf = v_inf
        self.v_star_surge = self.v_surge / self.v_inf
    
    def stitch_frames(self, frame0, frame1, x_trim = 0.3, z_trim = 0.025, frame_res=1.334/200, exp_factor = 12):
        """
        Stitch two frames together using overlapping weights. This method has only been tested for fields where the flow direction is left to right.
        Parameters:
        - frame0 (Frame): The first frame to be stitched. This will be the downstream frame. Unfortunatley, the naming convention is opposite to the actual flow direction due to Davis PIV software.
        - frame1 (Frame): The second frame to be stitched. This will be the upstream frame. If this convention is not followed, the overlap region will be incorrect.
        - frame_res (float): The resolution of the resulting frame. For the surge_disc_reprocess data of w-tunn campaign piv processing parameters are
            - pixeL-pitch = 5.995 mm
            - window_size = 32x32 pixels
            - overlap = 75%
            - frame_res = window_size * (1 - overlap) / pixel_pitch. It is 1.334 mm for this dataset. Change this value accordingly for other piv processing routines        
        - exp_factor (int, optional): The exponential factor used in the weight calculation. Defaults to 12.
        Returns:
        None
        """
        
        for name, value in inspect.getmembers(self): #Check if the stitched data already exists
            
            if name == "grid_x" or name == "grid_z":
                return
        
        x_min = min(frame0.grid_x.min(), frame1.grid_x.min()) + x_trim
        x_max = max(frame0.grid_x.max(), frame1.grid_x.max()) - x_trim
        x_arr = np.arange(x_min, x_max, frame_res)
        
        z_min = max(frame0.grid_z.min(), frame1.grid_z.min()) + z_trim
        z_max = min(frame0.grid_z.max(), frame1.grid_z.max()) - z_trim
        z_arr = np.arange(z_max, z_min, -frame_res)
        
        self.grid_x, self.grid_z = np.meshgrid(x_arr, z_arr, indexing='ij')
        self.disc_loc_indx = [np.absolute(self.grid_x[:,0]).argmin(), np.absolute(self.grid_z[0,:]).argmin()]
        
        x_ovlap_min = frame0.grid_x.min()
        x_ovlap_max = frame1.grid_x.max()
        
        grid_x_norm = (self.grid_x - x_ovlap_min) / (x_ovlap_max - x_ovlap_min)
        grid_ovlap_wts_frame1 = np.clip(1/(1 + np.exp(exp_factor * (grid_x_norm - 0.5))), 0, 1).round(5)
        grid_ovlap_wts_frame0 = np.clip(1/(1 + np.exp(-exp_factor * (grid_x_norm - 0.5))), 0, 1).round(5)
        
        for name, value in inspect.getmembers(frame0):
            
            if fnmatch.fnmatch(name, "grid*") and name != "grid_x" and name != "grid_z" and name != "grid_valid_cell" and not fnmatch.fnmatch(name, "*intpr"):
                
                stitched_data = getattr(frame0, name+"_intpr")(self.grid_x,self.grid_z) * grid_ovlap_wts_frame0 + getattr(frame1, name+"_intpr")(self.grid_x,self.grid_z) * grid_ovlap_wts_frame1
                
                setattr(self, name, stitched_data)
                
    def gen_data_intpr(self):
        """
        Generate interpolated data for the given object.
        This method checks if the interpolated data already exists for the object. If it does, the interpolation is skipped.
        Otherwise, it interpolates the dependent data using the NearestNDInterpolator.
        Parameters:
        - self: The object instance.
        Returns:
        - None
        """
                
        for name, value in inspect.getmembers(self):
            
            if fnmatch.fnmatch(name, "grid*intpr"): 
                return
        
        for name, value in inspect.getmembers(self):
            
            if fnmatch.fnmatch(name, "grid*") and name != "grid_x" and name != "grid_z" and name != "grid_valid_cell":
            
                setattr(self, name + "_intpr", NearestNDInterpolator((self.grid_x.flatten(), self.grid_z.flatten()), getattr(self, name).flatten()))
                
    def set_disc_mask(self, disc_mask_x_ustream=0.1, disc_mask_x_dstream=0.3, disc_mask_z_symm=0.7):
        """
        Sets the disc mask for the grid based on the specified upstream, downstream, and symmetry boundaries with respect to disc centre.
        The parameters are relative to the disc centre.
        The default parameters were found to be the best compromise between data loss and noise.
        This has been accomplished by analysing the reconstructed pressure field for different mask sizes.
        Parameters:
        disc_mask_x_ustream (float): The upstream boundary for the disc mask in the x-direction. Default is 0.1.
        disc_mask_x_dstream (float): The downstream boundary for the disc mask in the x-direction. Default is 0.3.
        disc_mask_z_symm (float): The symmetry boundary for the disc mask in the z-direction. Default is 0.7.
        Returns:
        None
        """
        
        self.disc_mask = np.logical_and(np.logical_and(self.grid_x >= -disc_mask_x_ustream, self.grid_x <= disc_mask_x_dstream), np.logical_and(self.grid_z >= -disc_mask_z_symm, self.grid_z <= disc_mask_z_symm))
                
    def gen_unshadow_field(self, disc_mask_x_ustream=0.1, disc_mask_x_dstream=0.3, disc_mask_z_symm=0.7, disc_dia=200e-3, mlab_eng_obj=mlab_eng1):
        """
        Generates unshadowed fields for velocity and Reynolds stress components by interpolating the mask region and smoothing the whole field.
        Unshadowing refers to filling the masked regions with interpolated values to remove the shadowing effect of the disc.
        Aditionally, this method smooths the interpolated values using MATLAB's smooth2 function. This is essential for accurate gradient calculations.
        This requires the disc mask to be set using the set_disc_mask method.
        Default way of generating unshadowed fields.
        Working MATLAB engine library is required for this method. Check https://www.mathworks.com/help/matlab/matlab-engine-for-python.html for installation instructions.
        Parameters:
        - disc_mask_x_ustream (float): Upstream x-coordinate mask boundary relative to the disc center.
        - disc_mask_x_dstream (float): Downstream x-coordinate mask boundary relative to the disc center.
        - disc_mask_z_symm (float): Symmetry z-coordinate mask boundary relative to the disc center.
        - disc_dia (float): Diameter of the disc in meters.
        - mlab_eng_obj: MATLAB engine object used for smoothing the interpolated values.
        Returns:
        - None: The function modifies the instance attributes by adding unshadowed fields.
        """
        
        for name, value in inspect.getmembers(self):
            
            if fnmatch.fnmatch(name, "unshdw*"): 
                return
            
        # Smaller outer disc region for efficient interpolation.
        outer_disc_mask = np.logical_and(~self.disc_mask, np.logical_and(self.grid_x >= -disc_mask_x_ustream-0.25, self.grid_x <= disc_mask_x_dstream+0.25))
            
        for name, value in inspect.getmembers(self):
            
            if name == "grid_v_x" or name == "grid_v_z" or name == "grid_r_xx" or name == "grid_r_xz" or name == "grid_r_zz":
            
                new_name = "unshdw_" + "_".join(name.split("_")[1:]) # Replace grid prefix with unshdw to indicate unshadowed field
                unshdw_field = deepcopy(value)
                
                # Apply user set mask and fill the mask with interpolated values.
                unshdw_field[self.disc_mask] = griddata((self.grid_x[outer_disc_mask], self.grid_z[outer_disc_mask]), unshdw_field[outer_disc_mask], (self.grid_x[self.disc_mask], self.grid_z[self.disc_mask]), method='linear')
                
                # Smoothing interpolated values using MATLAB's smooth2 function. Output of smooth2 is MATLAB double, hence converted to numpy array.
                # smooth2 function from MATLAB is used because it can handle boundary conditions better than scipy's uniform_filter.
                # Smoothing is done to remove the noise in the interpolated values. It is essential for accurate gradient calculations.
                unshdw_field = np.array(mlab_eng_obj.smooth2(unshdw_field, 7, 7))
                
                setattr(self, new_name, unshdw_field)
                
        self.unshdw_v_mag = np.sqrt(self.unshdw_v_x**2 + self.unshdw_v_z**2)
        self.unshdw_omega_y = np.gradient(self.unshdw_v_z, self.grid_x[:,0]*disc_dia, axis=0) - np.gradient(self.unshdw_v_x, self.grid_z[0,:]*disc_dia, axis=1)
        
    def get_unshadow_field(self, field_id, field_mask, x_ustr_indx, x_dstr_indx, mlab_eng_obj=mlab_eng1):
        """
        Generate an unshadowed field by interpolating and smoothing the data. This method allows the user to choose an arbitrary mask size.
        Recommended for testing and debugging purposes.
        Working MATLAB engine library is required for this method. Check https://www.mathworks.com/help/matlab/matlab-engine-for-python.html for installation instructions.
        Parameters:
        -----------
        field_id : str
            The attribute name of the field to be processed.
        field_mask : numpy.ndarray
            A boolean mask indicating the shadowed regions in the field.
        x_ustr_indx : int
            The upstream index in the x-direction for the grid.
        x_dstr_indx : int
            The downstream index in the x-direction for the grid.
        mlab_eng_obj : object, optional
            The MATLAB engine object used for smoothing the field. Default is mlab_eng1.
        Returns:
        --------
        numpy.ndarray
            The unshadowed field after interpolation and smoothing.
        """
        
        outer_mask = np.logical_and(~field_mask, np.logical_and(self.grid_x >= self.grid_x[x_ustr_indx,0] - 0.25, self.grid_x <= self.grid_x[x_dstr_indx,0] + 0.25))
        unshdw_field = deepcopy(getattr(self, field_id))
        unshdw_field[field_mask] = griddata((self.grid_x[outer_mask], self.grid_z[outer_mask]), unshdw_field[outer_mask], (self.grid_x[field_mask], self.grid_z[field_mask]), method='linear')
        unshdw_field = np.array(mlab_eng_obj.smooth2(unshdw_field, 7, 7))
        
        return unshdw_field
                
    def gen_pressure(self, p_inf, v_inf, rho_inf, mu=1.8e-5, disc_dia=200e-3, mlab_eng_obj=mlab_eng1):
        """
        Generate pressure field for a surging disc using velocity and Reynolds stress gradients.
        This method uses the matlab engine object to run Daniele Ragni's MATLAB code for solving the Poisson equation.
        This method requires the unshadowed fields to be generated using the gen_unshadow_field method.
        Working MATLAB engine library is required for this method. Check https://www.mathworks.com/help/matlab/matlab-engine-for-python.html for installation instructions.
        Parameters:
        -----------
        p_inf : float
            Free-stream pressure.
        v_inf : float
            Free-stream velocity.
        rho_inf : float
            Free-stream density.
        mu : float, optional
            Dynamic viscosity (default is 1.8e-5).
        disc_dia : float, optional
            Diameter of the disc (default is 200e-3).
        mlab_eng_obj : object, optional
            MATLAB engine object for executing MATLAB code (default is mlab_eng1).
        Returns:
        --------
        None
            The function updates the pressure field `self.grid_p` in place.
        """
        
        
        delta_x = self.grid_x[1,0] - self.grid_x[0,0]
        delta_z = self.grid_z[0,1] - self.grid_z[0,0]
        
        # Calcute velocity gradients
        unshdw_v_x_grad_x = np.gradient(self.unshdw_v_x, self.grid_x[:,0]*disc_dia, axis=0)
        unshdw_v_x_grad_z = np.gradient(self.unshdw_v_x, self.grid_z[0,:]*disc_dia, axis=1)
        unshdw_v_x_grad_xx = np.gradient(unshdw_v_x_grad_x, self.grid_x[:,0]*disc_dia, axis=0)
        unshdw_v_x_grad_xz = np.gradient(unshdw_v_x_grad_x, self.grid_z[0,:]*disc_dia, axis=1)
        unshdw_v_x_grad_zz = np.gradient(unshdw_v_x_grad_z, self.grid_z[0,:]*disc_dia, axis=1)
        
        unshdw_v_z_grad_x = np.gradient(self.unshdw_v_z, self.grid_x[:,0]*disc_dia, axis=0)
        unshdw_v_z_grad_z = np.gradient(self.unshdw_v_z, self.grid_z[0,:]*disc_dia, axis=1)
        unshdw_v_z_grad_xx = np.gradient(unshdw_v_z_grad_x, self.grid_x[:,0]*disc_dia, axis=0)
        unshdw_v_z_grad_xz = np.gradient(unshdw_v_z_grad_x, self.grid_z[0,:]*disc_dia, axis=1)
        unshdw_v_z_grad_zz = np.gradient(unshdw_v_z_grad_z, self.grid_z[0,:]*disc_dia, axis=1)
        
        # Calculate Reynolds stress gradients
        unshdw_r_xx_grad_x = np.gradient(self.unshdw_r_xx, self.grid_x[:,0]*disc_dia, axis=0)
        unshdw_r_xx_grad_z = np.gradient(self.unshdw_r_xx, self.grid_z[0,:]*disc_dia, axis=1)
        
        unshdw_r_xz_grad_x = np.gradient(self.unshdw_r_xz, self.grid_x[:,0]*disc_dia, axis=0)
        unshdw_r_xz_grad_z = np.gradient(self.unshdw_r_xz, self.grid_z[0,:]*disc_dia, axis=1)
        
        unshdw_r_zz_grad_x = np.gradient(self.unshdw_r_zz, self.grid_x[:,0]*disc_dia, axis=0)
        unshdw_r_zz_grad_z = np.gradient(self.unshdw_r_zz, self.grid_z[0,:]*disc_dia, axis=1)
        
        # Calculate pressure gradients
        # Alfonsi, Giancarlo. "Reynolds-averaged Navier–Stokes equations for turbulence modeling." Applied Mechanics Reviews 62.4 (2009).
        # Del Campo, V., et al. "3D load estimation on a horizontal axis wind turbine using SPIV." Wind Energy 17.11 (2014): 1645-1657.
        unshdw_p_grad_x = -rho_inf * (self.unshdw_v_x*unshdw_v_x_grad_x + self.unshdw_v_z*unshdw_v_x_grad_z) -rho_inf * (unshdw_r_xx_grad_x + unshdw_r_xz_grad_z) - rho_inf * self.unshdw_v_x_grad_t + mu * (unshdw_v_x_grad_xx + unshdw_v_x_grad_zz)
        unshdw_p_grad_z = -rho_inf * (self.unshdw_v_x*unshdw_v_z_grad_x + self.unshdw_v_z*unshdw_v_z_grad_z) -rho_inf * (unshdw_r_xz_grad_x + unshdw_r_zz_grad_z) - rho_inf * self.unshdw_v_z_grad_t + mu * (unshdw_v_z_grad_xx + unshdw_v_z_grad_zz)
        
        unshdw_p_grad_xx = np.gradient(unshdw_p_grad_x, self.grid_x[:,0]*disc_dia, axis=0)
        unshdw_p_grad_zz = np.gradient(unshdw_p_grad_z, self.grid_z[0,:]*disc_dia, axis=1)
        
        # Calculate Bernoulli pressure
        unshdw_p_bern = p_inf + 0.5 * rho_inf * (v_inf**2 - self.unshdw_v_mag**2)
        
        # Calculate pressure field using Daniele Ragni's MATLAB code for Poisson equation. MATLAB double output is converted to numpy array.
        # Disc mask convention in MATLAB code corresponds to the location of valid cells. Hence, the mask is inverted.
        # The data structure is transposed to match the MATLAB code's convention. The grid is transposed back to the original orientation after the calculation.
        # The last four parameters are the type of boundary conditions for the 2D Poisson equation. Boundary order convention is Left, Right, Top, Bottom.
        self.grid_p = np.array(mlab_eng_obj.Poisson_plane_nw(~self.disc_mask.T, unshdw_p_grad_x.T, unshdw_p_grad_xx.T, unshdw_p_grad_z.T, unshdw_p_grad_zz.T, unshdw_p_bern.T, delta_x*disc_dia, delta_z*disc_dia, 'Dir', 'Nx', 'Nx', 'Nx')).T
            
#%% Function to read PIV data from tecplot .dat files

def read_planar_piv_data(file_path):
    """
    Reads planar PIV data from a file and returns the data for two frames.
    frame0 is the downstream frame and frame1 is the upstream frame for my data. This ia an unfortunate naming convention due to Davis PIV software.
    Adapt the planar_piv_data and stitched_piv_data classes to your data if the naming convention is different.
    Parameters:
    - file_path (str): The path to the file containing the PIV data.
    Returns:
    - tuple: A tuple containing two planar_piv_data objects, one for each frame.
    """
    
    frame0_grid_info = open(file_path).readlines()[2].split(",")
    
    frame0_plane_i = int(frame0_grid_info[1].split("=")[1])
    frame0_plane_j = int(frame0_grid_info[2].split("=")[1])
    
    frame0_data = np.loadtxt(file_path, skiprows=4, max_rows=frame0_plane_i * frame0_plane_j)
    
    frame1_grid_info = open(file_path).readlines()[(frame0_plane_i * frame0_plane_j) + 5].split(",")
    
    frame1_plane_i = int(frame1_grid_info[1].split("=")[1])
    frame1_plane_j = int(frame1_grid_info[2].split("=")[1])
    
    frame1_data = np.loadtxt(file_path, skiprows=(frame0_plane_i * frame0_plane_j) + 7, max_rows=frame1_plane_i * frame1_plane_j)
    
    return planar_piv_data(frame0_plane_i, frame0_plane_j, frame0_data), planar_piv_data(frame1_plane_i, frame1_plane_j, frame1_data)

#%% Class to agglomerate multiple phase locked PIV runs of a case

class w_tunn_case_data:
    """
    A class to handle planar piv data from multiple phases of surge disc experiment held in w tunnel July 2023.
    Also it assimilates the test matrix and ambient conditions metadata into this object. 
    Do not use it on freestream fields. For analysing freestream fields directly import them using planar_piv_data class.
        case_uid (str): Unique identifier for the case.
        f_surge (float): Surge frequency.
        A_surge (float): Surge amplitude.
        omega_star_surge (float): Non-dimensional surge frequency.
        A_star_surge (float): Non-dimensional surge amplitude.
        T_surge (float): Surge period.
        dt_surge (float): Time step for surge.
        vmax_star_surge (float): Maximum non-dimensional surge velocity.
        phase_obj_list (list): List of phase objects containing processed data.
    Methods:
        __init__(case_df, phase_df, case_name, w_tunn_vinf_3_scaling=1.01048, w_tunn_vinf_2_scaling=1.01227):
            Initializes the w_tunn_case_data object with case and phase data.
        gen_grad_t_v2(disc_mask_x_ustream=0.05, disc_mask_x_dstream=0.25, disc_mask_z_symm=0.7, disc_dia=200e-3):
            Calculate time derivatives by creating a mask that encompasses the disc position from the previous to next phase. This method is suggested by Wei.
        gen_grad_t():
            Calculate time derivatives using the unshadowed gradient fields of the previous and next phases. This is the default method for calculating time derivatives.
    """
    
    def __init__(self, case_df, phase_df, case_name, w_tunn_vinf_3_scaling = 1.01048, w_tunn_vinf_2_scaling = 1.01227):
        """
        Initialize the w_tunn_case_data class.
        Parameters:
        case_df (pd.DataFrame): DataFrame containing case metadata. This dataframe can be made using provided case_metadata.csv file.
        phase_df (pd.DataFrame): DataFrame containing phase metadata. This dataframe can be made using provided phase_metadata.csv file.
        case_name (str): Unique identifier for the case. Valid case names are present in the case_df DataFrame.
        w_tunn_vinf_3_scaling (float, optional): Scaling factor for wind tunnel velocity (default is 1.01048). This is obtained by scaling the w-tunnel data v_inf with the corresponding freestream piv data.
        w_tunn_vinf_2_scaling (float, optional): Scaling factor for wind tunnel velocity (default is 1.01227).
        Attributes:
        case_uid (str): Unique identifier for the case.
        f_surge (float): Surge frequency.
        A_surge (float): Surge amplitude.
        omega_star_surge (float): Non-dimensional surge frequency.
        A_star_surge (float): Non-dimensional surge amplitude.
        T_surge (float): Surge period.
        dt_surge (float): Time step for surge.
        vmax_star_surge (float): Maximum non-dimensional surge velocity.
        phase_obj_list (list): List of phase objects.
        """
        
        # Assign case metadata
        self.case_uid = case_name
        case_metadata = case_df[case_df["case_uid"] == self.case_uid].values[0]
        self.f_surge, self.A_surge, self.omega_star_surge, self.A_star_surge = case_metadata[1:]
        if self.f_surge != 0:
            self.T_surge = 1/self.f_surge
            self.dt_surge = self.T_surge/10
        else:
            self.T_surge = np.nan # For static cases
        self.vmax_star_surge = self.A_star_surge * self.omega_star_surge
        
        case_root_path = data_dir + "/" + case_name
        phase_file_paths = sorted(glob.glob(case_root_path + "*.dat"))
        print(phase_file_paths)
        
        # List storing stitched_planar_piv_data objects for each surge motion phase.
        # For static cases, the list will contain only one object. For periodic cases, the list will contain multiple objects.
        # For static case, the fields are stored in phase_00 object.
        self.phase_obj_list = []
        
        for path in phase_file_paths:
        
            phase_uid = path.split("/")[-1].split(".")[0]
            fov_0, fov_1 = read_planar_piv_data(path)
            
            phase_metadata = phase_df[phase_df["phase_uid"] == phase_uid].values[0]
            
            # Exctracting phase metadata.
            fov_1_xc, fov_1_zc, fov_0_xcorr, fov_0_zcorr, piv_dt, v_surge, p_inf, rho_inf, w_tunn_v_inf = phase_metadata[1:]
            
            # Correcting the wind tunnel velocity using the scaling factors.
            if np.round(w_tunn_v_inf).astype(int) == 3.0:
                v_inf = w_tunn_v_inf * w_tunn_vinf_3_scaling
                
            elif np.round(w_tunn_v_inf).astype(int) == 2.0:
                v_inf = w_tunn_v_inf * w_tunn_vinf_2_scaling
            
            # align fov_0 with fov_1
            fov_0.corr_frame(fov_0_xcorr, fov_0_zcorr)
            
            # reorient both the fovs
            fov_0.reorient_frame(fov_1_xc, fov_1_zc)
            fov_1.reorient_frame(fov_1_xc, fov_1_zc)
            
            # generate field interpolators for both the frames
            fov_0.gen_data_intpr()
            fov_1.gen_data_intpr()
            
            stitched_fov = stitched_planar_piv_data(phase_uid, piv_dt, v_surge, p_inf, rho_inf, v_inf)
            stitched_fov.stitch_frames(fov_0, fov_1)
            stitched_fov.set_disc_mask(disc_mask_x_ustream=0.1, disc_mask_x_dstream=0.3, disc_mask_z_symm=0.7)
            stitched_fov.gen_unshadow_field()
            
            # Assign the stitched_piv object to the phase object
            setattr(self, "_".join(phase_uid.split("_")[3:]), stitched_fov)
            self.phase_obj_list.append(stitched_fov)
            
        # Generate time derivatives for the phase objects
        self.gen_grad_t()
        
        # Generate pressure fields for the phase objects
        for phase_obj in self.phase_obj_list:
            
            phase_obj.gen_pressure(phase_obj.p_inf, phase_obj.v_inf, phase_obj.rho_inf)
        
    def gen_grad_t_v2(self, disc_mask_x_ustream=0.05, disc_mask_x_dstream=0.25, disc_mask_z_symm=0.7, disc_dia=200e-3):
        """
        Generate temporal gradients for unshadowed fields.
        This method calculates the temporal gradients of various unshadowed fields
        for each phase object in the phase object list. The gradients are calculated
        based on the previous and next phase objects.
        Unshadowing here works differently than the default method gen_grad_t. 
        Super mask encompassing disc region of three phases involved in central difference derivative calculation is used for unshadowing.
        This is a self contained method. It does not require the disc mask to be set using set_disc_mask method and the unshadowed fields to be generated using gen_unshadow_field method.
        This method is suggested by Wei.
        Parameters:
        -----------
        disc_mask_x_ustream : float, optional
            The upstream distance for the disc mask in the x-direction (default is 0.05).
        disc_mask_x_dstream : float, optional
            The downstream distance for the disc mask in the x-direction (default is 0.25).
        disc_mask_z_symm : float, optional
            The symmetric distance for the disc mask in the z-direction (default is 0.7).
        disc_dia : float, optional
            The diameter of the disc (default is 200e-3).
        Returns:
        --------
        None
        """
        
        
        if self.f_surge == 0:
            
            for name, value in inspect.getmembers(self.phase_00):
            
                if fnmatch.fnmatch(name, "unshdw*grad_t"): 
                    return
            
            for name, value in inspect.getmembers(self.phase_00):
                
                if fnmatch.fnmatch(name, "unshdw*"):
                    
                    setattr(self, name + "_grad_t", np.zeros_like(value))
                    
        else:
            
            # Check if the gradient fields already exist
            for i, phase_obj in enumerate(self.phase_obj_list):
                
                for name, value in inspect.getmembers(phase_obj):
                        
                        if fnmatch.fnmatch(name, "unshdw*grad_t"): 
                            return
                
            for i, phase_obj in enumerate(self.phase_obj_list):
                
                # Get the previous and next phase objects
                if i < len(self.phase_obj_list) - 1:
                    
                    prev_phase_obj = self.phase_obj_list[i-1]
                    next_phase_obj = self.phase_obj_list[i+1]
                    
                else: # For the last phase object, the next phase object is the first phase object. Applicable only for periodic motions.
                    
                    prev_phase_obj = self.phase_obj_list[i-1]
                    next_phase_obj = self.phase_obj_list[0]
                    
                delta_x = phase_obj.grid_x[1,0] - phase_obj.grid_x[0,0]
                delta_z = phase_obj.grid_z[0,1] - phase_obj.grid_z[0,0]
                
                # Get the indices for the field mask. This mask corresponds to the regions where the disc surges in the current and adjacent phases.
                x_ustr_indx = np.min(np.array([prev_phase_obj.disc_loc_indx[0], phase_obj.disc_loc_indx[0], next_phase_obj.disc_loc_indx[0]])) - int(disc_mask_x_ustream//delta_x)
                x_dstr_indx = np.max(np.array([prev_phase_obj.disc_loc_indx[0], phase_obj.disc_loc_indx[0], next_phase_obj.disc_loc_indx[0]])) + int(disc_mask_x_dstream//delta_x)
                y_up_indx = np.max(np.array([prev_phase_obj.disc_loc_indx[1], phase_obj.disc_loc_indx[1], next_phase_obj.disc_loc_indx[1]])) + int(disc_mask_z_symm//delta_z)
                y_down_indx = np.min(np.array([prev_phase_obj.disc_loc_indx[1], phase_obj.disc_loc_indx[1], next_phase_obj.disc_loc_indx[1]])) - int(disc_mask_z_symm//delta_z)
                
                print(f"Loop index {i}:", x_ustr_indx, x_dstr_indx, y_up_indx, y_down_indx)
                
                field_mask = np.zeros_like(phase_obj.grid_x, dtype=bool)
                field_mask[x_ustr_indx:x_dstr_indx+1, y_up_indx:y_down_indx+1] = True
                phase_obj.field_mask = field_mask
                
                # Obtaining the unshadowed fields for the previous and next phases
                prev_unshdw_v_x = prev_phase_obj.get_unshadow_field("grid_v_x", field_mask, x_ustr_indx, x_dstr_indx)
                prev_unshdw_v_z = prev_phase_obj.get_unshadow_field("grid_v_z", field_mask, x_ustr_indx, x_dstr_indx)
                prev_unshdw_v_mag = np.sqrt(prev_unshdw_v_x**2 + prev_unshdw_v_z**2)
                prev_unshdw_omega_y = np.gradient(prev_unshdw_v_z, prev_phase_obj.grid_x[:,0]*disc_dia, axis=0) - np.gradient(prev_unshdw_v_x, prev_phase_obj.grid_z[0,:]*disc_dia, axis=1)
                prev_unshdw_r_xx = prev_phase_obj.get_unshadow_field("grid_r_xx", field_mask, x_ustr_indx, x_dstr_indx)
                prev_unshdw_r_xz = prev_phase_obj.get_unshadow_field("grid_r_xz", field_mask, x_ustr_indx, x_dstr_indx)
                prev_unshdw_r_zz = prev_phase_obj.get_unshadow_field("grid_r_zz", field_mask, x_ustr_indx, x_dstr_indx)
                
                next_unshdw_v_x = next_phase_obj.get_unshadow_field("grid_v_x", field_mask, x_ustr_indx, x_dstr_indx)
                next_unshdw_v_z = next_phase_obj.get_unshadow_field("grid_v_z", field_mask, x_ustr_indx, x_dstr_indx)
                next_unshdw_v_mag = np.sqrt(next_unshdw_v_x**2 + next_unshdw_v_z**2)
                next_unshdw_omega_y = np.gradient(next_unshdw_v_z, next_phase_obj.grid_x[:,0]*disc_dia, axis=0) - np.gradient(next_unshdw_v_x, next_phase_obj.grid_z[0,:]*disc_dia, axis=1)
                next_unshdw_r_xx = next_phase_obj.get_unshadow_field("grid_r_xx", field_mask, x_ustr_indx, x_dstr_indx)
                next_unshdw_r_xz = next_phase_obj.get_unshadow_field("grid_r_xz", field_mask, x_ustr_indx, x_dstr_indx)
                next_unshdw_r_zz = next_phase_obj.get_unshadow_field("grid_r_zz", field_mask, x_ustr_indx, x_dstr_indx)
                
                # Calculate the temporal gradients
                phase_obj.a_surge = (next_phase_obj.v_surge - prev_phase_obj.v_surge) / (2*self.dt_surge)
                phase_obj.unshdw_v_x_grad_t = (next_unshdw_v_x - prev_unshdw_v_x) / (2*self.dt_surge)
                phase_obj.unshdw_v_z_grad_t = (next_unshdw_v_z - prev_unshdw_v_z) / (2*self.dt_surge)
                phase_obj.unshdw_v_mag_grad_t = (next_unshdw_v_mag - prev_unshdw_v_mag) / (2*self.dt_surge)
                phase_obj.unshdw_omega_y_grad_t = (next_unshdw_omega_y - prev_unshdw_omega_y) / (2*self.dt_surge)
                phase_obj.unshdw_r_xx_grad_t = (next_unshdw_r_xx - prev_unshdw_r_xx) / (2*self.dt_surge)
                phase_obj.unshdw_r_xz_grad_t = (next_unshdw_r_xz - prev_unshdw_r_xz) / (2*self.dt_surge)
                phase_obj.unshdw_r_zz_grad_t = (next_unshdw_r_zz - prev_unshdw_r_zz) / (2*self.dt_surge)
                
                
    def gen_grad_t(self):
        """
        Generates the temporal gradient fields for the phase objects. Also calculates the surge acceleration.
        This method requires that the disc mask is set using the set_disc_mask method and the unshadowed fields are generated using the gen_unshadow_field method.
        This method seems to reliably reconstruct pressure field in comparison to the gen_grad_t_v2 method.
        Therefore this is the default method for calculating time derivatives.
        Attributes:
            f_surge (int): Flag indicating the surge condition.
            phase_00 (object): Phase object for the case when `f_surge` is 0.
            phase_obj_list (list): List of phase objects for the case when `f_surge` is not 0.
            dt_surge (float): Time step for surge calculation.
        """
        
        
        if self.f_surge == 0:
            
            for name, value in inspect.getmembers(self.phase_00):
            
                if fnmatch.fnmatch(name, "unshdw*grad_t"):
                    return
            
            for name, value in inspect.getmembers(self.phase_00):
                
                if fnmatch.fnmatch(name, "unshdw*"):
                    
                    setattr(self.phase_00, name + "_grad_t", np.zeros_like(value))
                    
        else:
            
            # Check if the gradient fields already exist
            for i, phase_obj in enumerate(self.phase_obj_list):
                
                for name, value in inspect.getmembers(phase_obj):
                        
                        if fnmatch.fnmatch(name, "unshdw*grad_t"): 
                            return
                
            for i, phase_obj in enumerate(self.phase_obj_list):
                
                if i < len(self.phase_obj_list) - 1:
                    
                    for name, value in inspect.getmembers(phase_obj):
                    
                        if fnmatch.fnmatch(name, "unshdw*"):
                        
                            setattr(phase_obj, name + "_grad_t", (getattr(self.phase_obj_list[i+1],name) - getattr(self.phase_obj_list[i-1],name)) / (2*self.dt_surge))
                            phase_obj.a_surge = (self.phase_obj_list[i+1].v_surge - self.phase_obj_list[i-1].v_surge) / (2*self.dt_surge)
                            
                else:
                    
                    for name, value in inspect.getmembers(phase_obj):
                    
                        if fnmatch.fnmatch(name, "unshdw*"):
                        
                            setattr(phase_obj, name + "_grad_t", (getattr(self.phase_obj_list[0],name)- getattr(self.phase_obj_list[i-1],name)) / (2*self.dt_surge))
                            phase_obj.a_surge = (self.phase_obj_list[0].v_surge - self.phase_obj_list[i-1].v_surge) / (2*self.dt_surge)
                                            

        
                            
#%% User input area. Testing.....
import time

w_tunn_case_name = "p45_case_07"
phase_df = pd.read_csv(data_dir + "/phase_metadata.csv")
case_df = pd.read_csv(data_dir + "/case_metadata.csv")

t0 = time.time()
w_tunn_case_obj = w_tunn_case_data(case_df, phase_df, w_tunn_case_name)
t1 = time.time()
print(f"Time taken for case data generation: {t1-t0} s")

#%% testing pickle dump and load
# pickle.dump(w_tunn_case_obj, open(data_dir + "/" + w_tunn_case_name + ".pkl", "wb"))
# w_tunn_case_obj = pickle.load(open(data_dir + "/" + w_tunn_case_name + ".pkl", "rb"))

#%% Time interpolator generation testing
t0 = time.time()
for phase_obj in w_tunn_case_obj.phase_obj_list:
    phase_obj.gen_data_intpr()
t1 = time.time()
print(f"Time taken for time interpolator generation: {t1-t0} s")

#%% Pressure plots
phase_obj_id = "phase_00"
field="grid_v_x"
plt.figure()
grid_x = getattr(w_tunn_case_obj, phase_obj_id).grid_x
grid_z = getattr(w_tunn_case_obj, phase_obj_id).grid_z
grid_field = masked_array(getattr(getattr(w_tunn_case_obj, phase_obj_id),field), getattr(w_tunn_case_obj, phase_obj_id).disc_mask)
grid_cp = (grid_field - getattr(w_tunn_case_obj, phase_obj_id).p_inf) / (0.5 * getattr(w_tunn_case_obj, phase_obj_id).rho_inf * getattr(w_tunn_case_obj, phase_obj_id).v_inf**2)
mapping_plot = plt.contourf(grid_x, grid_z, grid_field, levels=11, cmap='coolwarm')
# mapping_plot = plt.contourf(grid_x, grid_z, grid_cp, levels=11, cmap='coolwarm', norm=mpl.colors.TwoSlopeNorm(vcenter=0))
plt.colorbar(mapping_plot, orientation='horizontal')

# Plotting interpolated field
plt.figure()
grid_x_intpr = np.linspace(grid_x.min(), grid_x.max(), 500)
grid_z_intpr = np.linspace(grid_z.min(), grid_z.max(), 500)
grid_x_intpr, grid_z_intpr = np.meshgrid(grid_x_intpr, grid_z_intpr, indexing='ij')
grid_field_intpr = getattr(getattr(w_tunn_case_obj, phase_obj_id), field + "_intpr")(grid_x_intpr, grid_z_intpr)
# grid_field_intpr = masked_array(grid_field_intpr, getattr(w_tunn_case_obj, phase_obj_id).disc_mask)
grid_cp_intpr = (grid_field_intpr - getattr(w_tunn_case_obj, phase_obj_id).p_inf) / (0.5 * getattr(w_tunn_case_obj, phase_obj_id).rho_inf * getattr(w_tunn_case_obj, phase_obj_id).v_inf**2)
mapping_plot_intpr = plt.contourf(grid_x_intpr, grid_z_intpr, grid_field_intpr, levels=mapping_plot.levels, cmap='coolwarm')
# mapping_plot_intpr = plt.contourf(grid_x_intpr, grid_z_intpr, grid_cp_intpr, levels=mapping_plot.levels, cmap='coolwarm', norm=mpl.colors.TwoSlopeNorm(vcenter=0))
plt.colorbar(mapping_plot_intpr, orientation='horizontal')

#%% field mask testing
# w_tunn_case_obj.gen_grad_t_v2(0.1,0.3)
w_tunn_case_obj.gen_grad_t()

#%% plot field mask
plt.figure()
plt.title("Field mask testing")
plt.contourf(w_tunn_case_obj.phase_02.grid_x, w_tunn_case_obj.phase_02.grid_z, w_tunn_case_obj.phase_02.field_mask, levels=2, cmap='coolwarm')


#%% Testing unshdw field generation
plt1_phase = "phase_00"
plt2_phase = "phase_01"

plt.figure()
plt.title("Source v_x at preceding phase")
map_plot_src = plt.contourf(getattr(w_tunn_case_obj,plt1_phase).grid_x, getattr(w_tunn_case_obj,plt1_phase).grid_z, getattr(w_tunn_case_obj,plt1_phase).grid_v_x, levels=11, cmap='coolwarm')
plt.colorbar(map_plot_src, orientation='horizontal')
plt.gca().set_aspect('equal')

plt.figure()
plt.title("Time gradient of v_x at preceding phase")
map_plot = plt.contourf(getattr(w_tunn_case_obj,plt1_phase).grid_x, getattr(w_tunn_case_obj,plt1_phase).grid_z, getattr(w_tunn_case_obj,plt1_phase).unshdw_v_x_grad_t, levels=11, cmap='coolwarm')
plt.colorbar(map_plot, orientation='horizontal')


plt.figure()
plt.title("Source v_x at suceeding phase")
map_plot_src = plt.contourf(getattr(w_tunn_case_obj,plt2_phase).grid_x, getattr(w_tunn_case_obj,plt2_phase).grid_z, getattr(w_tunn_case_obj,plt2_phase).grid_v_x, levels=11, cmap='coolwarm')
plt.colorbar(map_plot_src, orientation='horizontal')
plt.gca().set_aspect('equal')

plt.figure()
plt.title("Derived v_x at suceeding phase")
# map_plot = plt.contourf(getattr(w_tunn_case_obj,plt1_phase).grid_x - (getattr(w_tunn_case_obj,plt1_phase).v_surge*w_tunn_case_obj.dt_surge + 0.5*getattr(w_tunn_case_obj,plt1_phase).a_surge*w_tunn_case_obj.dt_surge**2)/200e-3, getattr(w_tunn_case_obj,plt1_phase).grid_z, getattr(w_tunn_case_obj,plt1_phase).grid_v_x + getattr(w_tunn_case_obj,plt1_phase).unshdw_v_x_grad_t*w_tunn_case_obj.dt_surge, levels=map_plot_src.levels, cmap='coolwarm')
map_plot = plt.contourf(getattr(w_tunn_case_obj,plt2_phase).grid_x, getattr(w_tunn_case_obj,plt2_phase).grid_z, getattr(w_tunn_case_obj,plt1_phase).grid_v_x + getattr(w_tunn_case_obj,plt1_phase).unshdw_v_x_grad_t*w_tunn_case_obj.dt_surge, levels=map_plot_src.levels, cmap='coolwarm')
plt.colorbar(map_plot, orientation='horizontal')
plt.gca().set_aspect('equal')

# plt.figure()
# plt.title("Derived v_x from v_x_grad_t of smoothed filled v_x")
# vz_derived = getattr(w_tunn_case_obj,plt1_phase).grid_v_x + np.array(mlab_eng1.smooth2(getattr(w_tunn_case_obj,plt1_phase).unshdw_v_x_grad_t*w_tunn_case_obj.dt_surge, 7, 7))
# map_plot = plt.contourf(getattr(w_tunn_case_obj,plt1_phase).grid_x - (getattr(w_tunn_case_obj,plt1_phase).v_surge*w_tunn_case_obj.dt_surge + 0.5*getattr(w_tunn_case_obj,plt1_phase).a_surge*w_tunn_case_obj.dt_surge**2)/200e-3, getattr(w_tunn_case_obj,plt1_phase).grid_z, vz_derived, levels=map_plot_src.levels, cmap='coolwarm')
# plt.colorbar(map_plot, orientation="horizontal")
# plt.gca().set_aspect('equal')

#%%
type(w_tunn_case_obj.phase_00.unshdw_v_x)
type(w_tunn_case_obj.phase_00.unshdw_v_x_grad_t*w_tunn_case_obj.dt_surge)
print(w_tunn_case_obj.phase_00.v_surge * w_tunn_case_obj.dt_surge/200e-3)
print((w_tunn_case_obj.phase_00.v_surge*w_tunn_case_obj.dt_surge + 0.5*w_tunn_case_obj.phase_00.a_surge*w_tunn_case_obj.dt_surge**2)/200e-3)

# %%
dir(w_tunn_case_obj)
w_tunn_case_obj.phase_obj_list
print(w_tunn_case_obj.phase_00.p_inf, w_tunn_case_obj.phase_00.v_inf, w_tunn_case_obj.phase_00.rho_inf)
plt.figure()
map_plot = plt.contourf(masked_array(w_tunn_case_obj.phase_00.grid_x,w_tunn_case_obj.phase_00.disc_mask), masked_array(w_tunn_case_obj.phase_00.grid_z,w_tunn_case_obj.phase_00.disc_mask), masked_array((w_tunn_case_obj.phase_00.grid_p - w_tunn_case_obj.phase_00.p_inf)/(0.5*w_tunn_case_obj.phase_00.rho_inf*w_tunn_case_obj.phase_00.v_inf**2),w_tunn_case_obj.phase_00.disc_mask), levels=11, norm=mpl.colors.TwoSlopeNorm(vcenter=0), cmap='coolwarm')
plt.colorbar(map_plot, orientation='horizontal')
plt.gca().set_aspect('equal')
# %%
w_tunn_case_obj.phase_00.grid_p_intpr(0.26,-0.2)
# %%
len(w_tunn_case_obj.cyclic_phase_obj_list)
# %%
