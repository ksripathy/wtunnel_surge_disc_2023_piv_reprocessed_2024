#%%
import numpy as np
from numpy.ma import masked_array
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from copy import deepcopy
import inspect
import fnmatch
from scipy.interpolate import CubicSpline
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import uniform_filter
from scipy import signal
import pickle
from contourpy import contour_generator
from fractions import Fraction
import imageio
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap

#Configuring relative file locations
home_dir = os.path.abspath("")
plot_dir = os.path.join(home_dir,"plots")
data_dir = os.path.join(home_dir,"pickled_data")

# Define the colors for the custom colormap with more vibrant colors
colors = [
    (0.0, 0.0, 0.5),  # Dark Blue
    (0.0, 0.5, 1.0),  # Bright Blue
    (1.0, 1.0, 1.0),  # White
    (1.0, 0.5, 0.0),  # Bright Orange
    (0.5, 0.0, 0.0)   # Dark Red
]

# Create the custom colormap
vibrant_coolwarm = LinearSegmentedColormap.from_list('vibrant_coolwarm', colors, N=256)

from pack_data_libs import w_tunn_case_data, stitched_planar_piv_data

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

# Calculate percentiles using numpy.ma.median

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
    
#%% Class pickle data
class stitched_planar_piv_data_reconstruct:
    """A class to reconstruct and analyze PIV frame data of individual phases
    of wtunn surge disc 2023 exp, stored using stitched_planar_piv_data class.
    Attributes:
    grid_turb_intensity : numpy.ndarray
        The grid turbulence intensity calculated from the velocity components and their standard deviations.
    Methods:
    __init__(self, stitched_planar_piv_data_obj):
        Initializes the class by copying attributes from the given stitched_planar_piv_data_obj and calculates grid turbulence intensity.
    gen_data_intpr(self):
        Generates interpolated data for the object using NearestNDInterpolator if it does not already exist.
    get_mask(self, disc_mask_x_ustream=0.1, disc_mask_x_dstream=0.3, disc_mask_z_symm=0.7):
        Generates a mask based on the specified upstream, downstream, and symmetry limits.
    update_omega_y(self):
        Updates the omega_y attribute by calculating the gradient of the masked velocity components.
    get_unshdw_eff_ind(self, x, v_ref=1, mask_x_ustream=0.1, mask_x_dstream=0.3, mask_z_symm=0.51):
        Calculates the unshadowed effective induced velocity at a given x-location.
    get_eff_ind(self, x, v_ref):
        Calculates the effective induced velocity at a given x-location.
    get_eff_ind_intpr(self, v_ref, x_ustream_fin=-0.1, x_dstream_init=0.3, smooth=True):
        Interpolates the effective induced velocity over a specified range of x-coordinates using a cubic spline.
    get_avg_turb_intensity(self):
        Calculates the average turbulence intensity over a specified grid region.
    """
    
    def __init__(self, stitched_planar_piv_data_obj):
        """
        Initializes the class by copying attributes from the given stitched_planar_piv_data_obj and calculates grid turbulence intensity.
        Parameters:
        - stitched_planar_piv_data_obj: The object instance of stitched_planar_piv_data.
        Returns:
        - None
        """
        
        attrb_list = [attrb for attrb in dir(stitched_planar_piv_data_obj) if not callable(getattr(stitched_planar_piv_data_obj, attrb)) and not attrb.startswith("__")]
        
        for attrb in attrb_list:
            
            setattr(self, attrb, getattr(stitched_planar_piv_data_obj, attrb))
            
        self.grid_turb_intensity = np.sqrt(self.grid_v_x**2*self.grid_std_v_x**2 + self.grid_v_z**2*self.grid_std_v_z**2) / np.sqrt(self.grid_v_x**2 + self.grid_v_z**2)
            
    def gen_data_intpr(self):
        """
        Generate interpolated data for the given objecst.
        This method checks if the interpolated data already exists for the object. If it does, the interpolation is skipped.
        Otherwise, it interpolates the dependent data using the NearestNDInterpolator.
        Parameters:
        - self: The object instance.
        Returns:
        - None
        """
                
        for name, value in inspect.getmembers(self):
            
            if fnmatch.fnmatch(name, "grid*intpr"):
                print(f"Interpolated data already exists for {name}. Skipping interpolation.") 
                return
        
        for name, value in inspect.getmembers(self):
            
            if fnmatch.fnmatch(name, "grid*") and name != "grid_x" and name != "grid_z" and name != "grid_valid_cell":
            
                setattr(self, name + "_intpr", NearestNDInterpolator((self.grid_x.flatten(), self.grid_z.flatten()), getattr(self, name).flatten()))
        
    def get_mask(self, disc_mask_x_ustream=0.1, disc_mask_x_dstream=0.3, disc_mask_z_symm=0.7):
        """
        Generate a mask based on specified upstream, downstream, and symmetry limits.
        This is based on the set_disc_mask method in the stitched_planar_piv_data class.
        Here, the obtained mask need not enclose the entire disc region.
        Multiple masks of different sizes can be generated based on the specified limits.
        It is useful for precisely masking the disc region instead of just using the conservative disc_mask.
        Parameters:
        disc_mask_x_ustream (float): The upstream limit for the mask in the x-direction. Default is 0.1.
        disc_mask_x_dstream (float): The downstream limit for the mask in the x-direction. Default is 0.3.
        disc_mask_z_symm (float): The symmetry limit for the mask in the z-direction. Default is 0.7.
        Returns:
        numpy.ndarray: A boolean array where the mask is True within the specified limits and False otherwise.
        """
        
        
        return np.logical_and(np.logical_and(self.grid_x >= -disc_mask_x_ustream, self.grid_x <= disc_mask_x_dstream), np.logical_and(self.grid_z >= -disc_mask_z_symm, self.grid_z <= disc_mask_z_symm))
    
    def update_omega_y(self):
        """
        Updates the `updtd_omega_y` attribute by calculating the vorticity component in the y-direction.
        This method applies masks to the velocity fields `grid_v_x` and `grid_v_z` to exclude certain regions
        and then computes the y-component of the vorticity using the gradients of the masked velocity fields.
        The intention of this method is to recalculate vorticity after excluding the regions around the disc
        instead of using DaVis vorticity field, which was obtained without masking.
        The masks are defined as follows:
        - `ustream_mask`: Mask for the upstream region.
        - `main_mask`: Mask for the main region.
        - `dstream_mask`: Mask for the downstream region.
        - `disc_mask`: Combined mask of the upstream, main, and downstream regions.
        The vorticity component in the y-direction is calculated using the following formula:
        `updtd_omega_y = ∂(masked_v_z)/∂x - ∂(masked_v_x)/∂z`
        The gradients are computed with respect to the grid coordinates scaled by a factor of 200e-3.
        Returns:
            None
        """
                
        ustream_mask = self.get_mask(0.1,-0.06,0.04)
        main_mask = self.get_mask(0.06,0.15,0.55)
        dstream_mask = self.get_mask(-0.15,0.3,0.06)
        disc_mask = np.logical_or(np.logical_or(ustream_mask, main_mask), dstream_mask)
        
        masked_v_x = masked_array(self.grid_v_x, mask=disc_mask)
        masked_v_z = masked_array(self.grid_v_z, mask=disc_mask)
        self.updtd_omega_y = np.gradient(masked_v_z, self.grid_x[:,0]*200e-3, axis=0) - np.gradient(masked_v_x, self.grid_z[0,:]*200e-3, axis=1)
        
    def get_unshdw_eff_ind(self, x, v_ref=1, mask_x_ustream=0.1, mask_x_dstream=0.3, mask_z_symm=0.51):
        """
        Calculate the unshadowed effective induced velocity at a given x-coordinate.
        This method is based on ANdrea's suggestion to perform unshadowing excluding the regions above and below the disc 
        thereby eliminating the effect of acceleration effects above and below the disc.
        This is done by considering the same vertical limits for the disc_mask and the outer_mask.
        Default mask_z_symm=0.51 ensures that these acceleration effects are excluded.
        Parameters:
        x (float): The x-coordinate at which to calculate the effective induced velocity.
        v_ref (float, optional): Reference velocity for normalization. Default is 1.
        mask_x_ustream (float, optional): Upstream x-coordinate for the mask. Default is 0.1.
        mask_x_dstream (float, optional): Downstream x-coordinate for the mask. Default is 0.3.
        mask_z_symm (float, optional): Symmetry z-coordinate for the mask. Default is 0.51.
        Returns:
        float: The unshadowed effective induced velocity.
        """
        
        mask = self.get_mask(mask_x_ustream, mask_x_dstream, mask_z_symm)
        outer_mask = np.logical_and(self.get_mask(mask_x_ustream+0.25, mask_x_dstream+0.25, mask_z_symm), np.logical_not(mask))
        
        # Plotting the masks. Intended for debugging purposes.
        # plt.figure()
        # plt.contourf(self.grid_x, self.grid_z, mask, levels=2, cmap='coolwarm')
        # plt.title("Mask")
        
        # plt.figure()
        # plt.contourf(self.grid_x, self.grid_z, outer_mask, levels=2, cmap='coolwarm')
        # plt.title("Outer Mask")
        
        delta_z = np.absolute(self.grid_z[0,1] - self.grid_z[0,0])
        z_arr = np.arange(-0.5,0.5+delta_z,delta_z)
        radii_arr = np.absolute(z_arr)
        x_arr = np.ones_like(z_arr) * x
        v_x_arr = griddata((self.grid_x[outer_mask], self.grid_z[outer_mask]), self.grid_v_x[outer_mask], (x_arr, z_arr), method='linear')
        ind_arr = (self.v_inf - v_x_arr) / v_ref
        
        eff_ind = np.trapezoid(ind_arr * radii_arr, z_arr) / np.trapezoid(radii_arr, z_arr)
        
        # Plotting the unshadowed velocity field. Intended for debugging purposes.
        # v_x = deepcopy(self.grid_v_x)
        # v_x[mask] = griddata((self.grid_x[outer_mask], self.grid_z[outer_mask]), self.grid_v_x[outer_mask], (self.grid_x[mask], self.grid_z[mask]), method='linear')
        
        # plt.figure()
        # map_plot = plt.contourf(self.grid_x, self.grid_z, v_x/self.v_inf, levels=np.linspace(0.5,1.2,8), norm=mpl.colors.TwoSlopeNorm(vmin=0.5,vcenter=1,vmax=1.2), cmap='coolwarm')
        # plt.title("Unshadowed v_x new")
        # plt.colorbar(map_plot, orientation='horizontal')
        
        # plt.figure()
        # map_plot = plt.contourf(self.grid_x, self.grid_z, self.grid_v_x/self.v_inf, levels=np.linspace(0.5,1.2,8), norm=mpl.colors.TwoSlopeNorm(vmin=0.5,vcenter=1,vmax=1.2), cmap='coolwarm')
        # plt.title("Grid v_x")
        # plt.colorbar(map_plot, orientation='horizontal')
        
        return eff_ind
                
    def get_eff_ind(self, x, v_ref):
        """
        Calculate the effective induced velocity at a given x-location.
        Parameters:
        x (float): The x-location where the effective induced velocity is calculated.
        v_inf (float): The free-stream velocity.
        Returns:
        float: The effective induced velocity at the specified x-location.
        """
        
        delta_z = np.absolute(self.grid_z[0,1] - self.grid_z[0,0])        
        
        y2_loc = 0.5
        y1_loc = -0.5
        
        y_arr = np.arange(y1_loc, y2_loc+delta_z, delta_z)
        radii_arr = np.absolute(y_arr)
                
        v_x_arr = self.grid_v_x_intpr(x, y_arr)
        ind_arr = (self.v_inf - v_x_arr) / v_ref
        
        eff_ind = np.trapezoid(ind_arr * radii_arr, y_arr) / np.trapezoid(radii_arr, y_arr)
        
        return eff_ind
    
    def get_eff_ind_intpr(self, v_ref, x_ustream_fin=-0.1, x_dstream_init=0.3, smooth=True):
        """
        Interpolates the effective induced velocity over a specified range of x-coordinates.
        Parameters:
        -----------
        v_inf : float
            The free-stream velocity.
        x_ustream_fin : float, optional
            The final x-coordinate upstream of the object (default is -0.1).
        x_dstream_init : float, optional
            The initial x-coordinate downstream of the object (default is 0.2).
        smooth : bool, optional
            If True, applies a low-pass filter to smooth the effective induced velocity arrays (default is True) enhancing the effectiveness of interpolation.
        Returns:
        --------
        CubicSpline
            A cubic spline interpolation of the effective induced velocity over the concatenated upstream and downstream x-coordinates.
        """
        
        
        delta_x = self.grid_x[1,0] - self.grid_x[0,0]
        x_ustream_arr = np.arange(-0.5,x_ustream_fin,delta_x)
        x_dstream_arr = np.arange(x_dstream_init,self.grid_x.max(),delta_x)
        
        eff_ind_ustream_arr = np.array([self.get_eff_ind(x, v_ref) for x in x_ustream_arr])
        eff_ind_dstream_arr = np.array([self.get_eff_ind(x, v_ref) for x in x_dstream_arr])
        
        if smooth:
        
            eff_ind_ustream_arr = low_pass_filt(eff_ind_ustream_arr, 1/delta_x, 0.05/delta_x)
            eff_ind_dstream_arr = low_pass_filt(eff_ind_dstream_arr, 1/delta_x, 0.05/delta_x)
        
        return CubicSpline(np.concatenate((x_ustream_arr, x_dstream_arr)), np.concatenate((eff_ind_ustream_arr, eff_ind_dstream_arr)))
    
    def get_avg_turb_intensity(self):
        """
        Calculate the average turbulence intensity over a specified grid.
        This method computes the average turbulence intensity by interpolating
        the turbulence intensity values over a grid defined by the x and z 
        coordinates. The grid is created with specified ranges and intervals 
        based on the differences between consecutive grid points in the x and z 
        directions.
        Returns:
            float: The average turbulence intensity over the specified grid.
        """
        
        
        delta_x = np.absolute(self.grid_x[1,0] - self.grid_x[0,0])
        delta_z = np.absolute(self.grid_z[0,1] - self.grid_z[0,0])
        
        x_arr = np.arange(0.5,1,delta_x)
        z_arr = np.arange(-0.25,0.25,delta_z)
        
        grid_x, grid_z = np.meshgrid(x_arr, z_arr, indexing='ij')
        
        return np.mean(self.grid_turb_intensity_intpr(grid_x.flatten(), grid_z.flatten()))

class w_tunn_case_data_reconstruct:
    """
    A class to reconstruct wind tunnel case data and perform various analyses on it.
    Attributes:
    -----------
    phase_obj_list : list
        A list of phase objects reconstructed from the wind tunnel case object.
    phi_all_phases : numpy.ndarray
        Array of phase angles for all phases.
    v_inf_all_phases : numpy.ndarray
        Array of free-stream velocities for all phases.
    p_inf_all_phases : numpy.ndarray
        Array of free-stream pressures for all phases.
    rho_inf_all_phases : numpy.ndarray
        Array of free-stream densities for all phases.
    v_surge_all_phases : numpy.ndarray
        Array of surge velocities for all phases.
    v_app_all_phases : numpy.ndarray
        Array of apparent velocities for all phases.
    eff_ind_ground_all_phases : numpy.ndarray
        Array of effective induced velocities at the ground for all phases.
    eff_ind_disc_all_phases : numpy.ndarray
        Array of effective induced velocities at the disc for all phases.
    unshdw_eff_ind_ground_all_phases : numpy.ndarray
        Array of unshadowed effective induced velocities at the ground for all phases.
    unshdw_eff_ind_disc_all_phases : numpy.ndarray
        Array of unshadowed effective induced velocities at the disc for all phases.
    avg_turb_intensity_all_phases : numpy.ndarray
        Array of average turbulence intensities for all phases.
    Methods:
    --------
    __init__(w_tunn_case_obj):
        Initializes the w_tunn_case_data_reconstruct object by copying attributes from the wind tunnel case object and reconstructing phase objects.
    get_phase_obj_list_attrs(attribute, args=()):
        Returns the requested attribute of all the objects in the phase_obj_list.
    export_field_gif(field_attrb):
        Exports a GIF of the specified field attribute for all phases.
    """
    
    
    def __init__(self, w_tunn_case_obj):
        """
        Initializes the analysis object with wind tunnel case data.
        Parameters:
        w_tunn_case_obj (object): An object containing wind tunnel case data.
        Attributes:
        phase_obj_list (list): A list of phase objects reconstructed from the wind tunnel case data.
        phi_all_phases (ndarray): Array of phase angles for all phases.
        v_inf_all_phases (ndarray): Array of free-stream velocities for all phases.
        p_inf_all_phases (ndarray): Array of free-stream pressures for all phases.
        rho_inf_all_phases (ndarray): Array of free-stream densities for all phases.
        v_surge_all_phases (ndarray): Array of surge velocities for all phases.
        v_app_all_phases (ndarray): Array of apparent velocities for all phases.
        eff_ind_ground_all_phases (ndarray): Array of effective induced velocities at the ground for all phases.
        eff_ind_disc_all_phases (ndarray): Array of effective induced velocities at the disc for all phases.
        unshdw_eff_ind_ground_all_phases (ndarray): Array of unshadowed effective induced velocities at the ground for all phases.
        unshdw_eff_ind_disc_all_phases (ndarray): Array of unshadowed effective induced velocities at the disc for all phases.
        avg_turb_intensity_all_phases (ndarray): Array of average turbulence intensities for all phases.
        """
        
        
        metadata_attrb_list = [attrb for attrb in dir(w_tunn_case_obj) if not callable(getattr(w_tunn_case_obj, attrb)) and not attrb.startswith("__") and not attrb.startswith("phase")]
        
        for attrb in metadata_attrb_list:
            
            setattr(self, attrb, getattr(w_tunn_case_obj, attrb))
            
        phase_obj_attrb_list = [attrb for attrb in dir(w_tunn_case_obj) if not callable(getattr(w_tunn_case_obj, attrb)) and not attrb.startswith("__") and not attrb.startswith("phase_obj_list") and attrb.startswith("phase")]
        self.phase_obj_list = []
        
        for attrb in phase_obj_attrb_list:
            
            setattr(self,attrb,stitched_planar_piv_data_reconstruct(getattr(w_tunn_case_obj, attrb)))
            self.phase_obj_list.append(getattr(self,attrb))
            
        # Generate interpolators for all the phase objects
        self.get_phase_obj_list_attrs("gen_data_intpr")
        
        # Update omega_y for all the phase objects
        self.get_phase_obj_list_attrs("update_omega_y")
        
        # Collecting metadata for all the phases
        self.phi_all_phases = np.linspace(0, 2*np.pi, len(self.phase_obj_list), endpoint=False)
        self.v_inf_all_phases = self.get_phase_obj_list_attrs("v_inf")
        self.p_inf_all_phases = self.get_phase_obj_list_attrs("p_inf")
        self.rho_inf_all_phases = self.get_phase_obj_list_attrs("rho_inf")
        self.v_surge_all_phases = self.get_phase_obj_list_attrs("v_surge")
        self.v_app_all_phases = self.v_inf_all_phases - self.v_surge_all_phases
        
        # Deriving effective induced velocities for all the phases for different reference frames
        # Ground reference frame uses vref = v_inf, disc reference frame uses vref = v_app
        self.eff_ind_ground_all_phases = np.array([phase_obj.get_eff_ind_intpr(v_ref=v_inf)(0) for phase_obj, v_inf in zip(self.phase_obj_list, self.v_inf_all_phases)])
        self.eff_ind_disc_all_phases = np.array([phase_obj.get_eff_ind_intpr(v_ref=v_app)(0) for phase_obj, v_app in zip(self.phase_obj_list, self.v_app_all_phases)])
        self.unshdw_eff_ind_ground_all_phases = np.array([phase_obj.get_unshdw_eff_ind(0, v_ref=v_inf) for phase_obj, v_inf in zip(self.phase_obj_list, self.v_inf_all_phases)])
        self.unshdw_eff_ind_disc_all_phases = np.array([phase_obj.get_unshdw_eff_ind(0, v_ref=v_app) for phase_obj, v_app in zip(self.phase_obj_list, self.v_app_all_phases)])
        self.avg_turb_intensity_all_phases = self.get_phase_obj_list_attrs("get_avg_turb_intensity")

    def get_phase_obj_list_attrs(self, attribute, args=()): 
        """Returns the requested attribute of all the objects in the phase_obj_list."""
        
        if not callable (getattr(self.phase_00,attribute)):
            # If the requested attribute is a variable
            return np.array([getattr(phase_obj, attribute) for phase_obj in self.phase_obj_list])
        else:
            # If the requested attribute is a method
            return np.array([getattr(obj, attribute)(*args) for obj in self.phase_obj_list])
        
    def export_field_gif(self, field_attrb):
        """
        Exports a GIF animation of a specified field attribute over all phases.
        Parameters:
        -----------
        field_attrb : str
            The field attribute to be visualized and exported as a GIF. 
            Possible values include "*v_mag" for velocity magnitude, "*omega_y" for vorticity, and "*p" for pressure.
        Description:
        ------------
        This method generates a GIF animation of the specified field attribute over all phases. 
        It first obtains the necessary field data and coordinates, applies appropriate masks, and non-dimensionalizes the data if required. 
        It then creates contour plots for each phase, optionally adding quiver plots for velocity fields. 
        Finally, it compiles these plots into a GIF and saves it to the specified directory.
        The GIF includes:
        - Contour plots of the specified field attribute.
        - Quiver plots for velocity fields if the attribute is "*v_mag".
        - Titles indicating the disc case, surge parameters, and phase angle.
        The method handles three types of field attributes:
        - Velocity magnitude ("*v_mag"): Non-dimensionalized by free-stream velocity.
        - Pressure ("*p"): Non-dimensionalized by dynamic pressure.
        - Vorticity ("*omega_y"): Scaled by disc diameter and free-stream velocity.
        The resulting GIF is saved with a filename format of "{case_uid}_{field_attrb}.gif".
        Raises:
        -------
        ValueError:
            If the field attribute does not match any of the expected patterns.
        """
        
        
        #Obtain an array of field data coordinates for all the phases
        plot_x_all_phases = self.get_phase_obj_list_attrs("grid_x")
        plot_z_all_phases = self.get_phase_obj_list_attrs("grid_z")
        
        # Small mask for vel and vort fields. Large mask for pressure field
        if fnmatch.fnmatch(field_attrb, "*v_mag") or fnmatch.fnmatch(field_attrb, "*omega_y"):
            
            ustream_mask_all_phases = self.get_phase_obj_list_attrs("get_mask",(0.1,-0.075,0.04))
            main_mask_all_phases = self.get_phase_obj_list_attrs("get_mask",(0.075,0.15,0.6))
            dstream_mask_all_phases = self.get_phase_obj_list_attrs("get_mask",(-0.15,0.3,0.075))
            disc_mask_all_phases = np.logical_or(np.logical_or(ustream_mask_all_phases, main_mask_all_phases), dstream_mask_all_phases)
            
        elif fnmatch.fnmatch(field_attrb, "*p"):
            
            disc_mask_all_phases = self.get_phase_obj_list_attrs("disc_mask")
        
        # Obtain an array of field data for all the phases
        field_data_all_phases = self.get_phase_obj_list_attrs(field_attrb)
        masked_data_all_phases = np.ma.array([masked_array(data, mask=disc_mask) for data, disc_mask in zip(field_data_all_phases, disc_mask_all_phases)]) # Use np.ma.array to preserve the mask
        
        # Generate non-dimensionalized field data
        if fnmatch.fnmatch(field_attrb, "*v_mag"):
            
            plot_data_all_phases = masked_data_all_phases / self.v_inf_all_phases[:,np.newaxis,np.newaxis] # denominator is broadcasted to the shape of the masked_data_all_phases. This is done to perform element-wise division.
            plot_vcentre = 1
            plot_label = "v_mag/v_inf [-]"
            plot_extend = "neither"
            # min_plot_data = np.percentile(plot_data_all_phases.compressed(), 0.01)
            min_plot_data = np.min(plot_data_all_phases)
            max_plot_data = np.max(plot_data_all_phases)
            
            # Generate quivers for velocity fields
            v_x_all_phases = self.get_phase_obj_list_attrs("grid_v_x")
            v_z_all_phases = self.get_phase_obj_list_attrs("grid_v_z")
            masked_v_x_all_phases = np.ma.array([masked_array(v_x, mask) for v_x,mask in zip(v_x_all_phases,disc_mask_all_phases)]) / self.v_inf_all_phases[:,np.newaxis,np.newaxis]
            masked_v_z_all_phases = np.ma.array([masked_array(v_z, mask) for v_z,mask in zip(v_z_all_phases,disc_mask_all_phases)]) / self.v_inf_all_phases[:,np.newaxis,np.newaxis]
        
            
        elif fnmatch.fnmatch(field_attrb, "*p"):
            
            plot_data_all_phases = (masked_data_all_phases - self.p_inf_all_phases[:,np.newaxis,np.newaxis]) / (0.5 * self.rho_inf_all_phases[:,np.newaxis,np.newaxis] * self.v_inf_all_phases[:,np.newaxis,np.newaxis]**2)
            plot_vcentre = 0
            plot_label = "Cp [-]"
            plot_extend = "neither"
            min_plot_data = np.min(plot_data_all_phases)
            max_plot_data = np.max(plot_data_all_phases)
            
        elif fnmatch.fnmatch(field_attrb, "*omega_y"):
            
            plot_data_all_phases = masked_data_all_phases * 200e-3 / self.v_inf_all_phases[:,np.newaxis,np.newaxis]
            plot_vcentre = 0
            plot_label = "omega_y * D/v_inf [-]"
            plot_extend = "both"
            min_plot_data = np.min(plot_data_all_phases)/5 # Good compromise between resolution and visibility
            max_plot_data = np.max(plot_data_all_phases)/5
            
        # Generate plot masks for all the phases
        
        plot_mask_all_phases = np.ma.array([masked_array(data, mask=np.logical_not(data)) for data in disc_mask_all_phases])

        figs_all_phases = [plt.figure() for _ in range(len(self.phase_obj_list))]
        
        for fig, plot_x, plot_z, plot_data in zip(figs_all_phases, plot_x_all_phases, plot_z_all_phases, plot_data_all_phases):
            
            ax = fig.add_subplot(111)
            mask_plot = ax.contourf(plot_x, plot_z, plot_mask_all_phases[figs_all_phases.index(fig)], colors="gray")
            field_plot = ax.contourf(plot_x, plot_z, plot_data, levels=np.linspace(min_plot_data,max_plot_data,11), norm=mpl.colors.TwoSlopeNorm(vmin=min_plot_data,vcenter=plot_vcentre,vmax=max_plot_data,), cmap=vibrant_coolwarm, extend=plot_extend) # Custom colormap with 256 levels is used to avoid crushingin gifs
            if fnmatch.fnmatch(field_attrb, "*v_mag"):
                quiver_plot = ax.quiver(plot_x[::32,::32], plot_z[::32,::32], masked_v_x_all_phases[figs_all_phases.index(fig)][::32,::32], masked_v_z_all_phases[figs_all_phases.index(fig)][::32,::32], color='k')
            fig.colorbar(field_plot, orientation='horizontal', label=plot_label, format='%.1f')
            ax.set_aspect('equal')
            ax.set_xlabel('x/D [-]')
            ax.set_ylabel('z/D [-]')
            ax.set_title(f"Disc-{self.case_uid.split("_")[0]}, $\omega^*$ = {self.omega_star_surge}, $A^*$ = {self.A_star_surge}, " + f"$\phi$ = {Fraction(self.phi_all_phases[figs_all_phases.index(fig)]/np.pi).limit_denominator()}$\pi$ rad") # Add phase angle to the title.
            
        # Export the figures to a gif
        gif_images = []
        for fig in figs_all_phases:
            buf = BytesIO() # Create a buffer to store the image. THis means that the image is saved in memory and not on disk.
            fig.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            gif_images.append(imageio.imread(buf))
            buf.close()
        
        gif_filename = os.path.join(plot_dir, f"{self.case_uid}_{field_attrb}.gif")
        imageio.mimsave(gif_filename, gif_images, fps=1)
            
#%% Static cases
case_06_obj = w_tunn_case_data_reconstruct(pickle.load(open(os.path.join(data_dir,"p45_case_06.pkl"), "rb")))
case_07_obj = w_tunn_case_data_reconstruct(pickle.load(open(os.path.join(data_dir,"p45_case_07.pkl"), "rb")))

#%% Testing
case_06_obj.get_phase_obj_list_attrs("get_unshdw_eff_ind", (0.1,0.3,0.55))            


#%% Testing
test_obj = w_tunn_case_data_reconstruct(pickle.load(open(os.path.join(data_dir,"p45_case_05.pkl"), "rb")))  

#%% Testing
test_obj.get_phase_obj_list_attrs("get_unshdw_eff_ind", (0.1,0.3,0.51))   

#%% Testing export_field_gif
test_obj.export_field_gif("grid_v_mag")
test_obj.export_field_gif("grid_omega_y")

#%% Grid mask all phases test
plt.figure()
plt.contourf(test_obj.phase_00.grid_x, test_obj.phase_00.grid_z, test_obj.phase_00.disc_mask, levels=2, cmap='coolwarm')

disc_mask_arr = test_obj.get_phase_obj_list_attrs("disc_mask")
plt.figure()
plt.contourf(test_obj.phase_00.grid_x, test_obj.phase_00.grid_z, disc_mask_arr[0], levels=2, cmap='coolwarm')

data_arr = test_obj.get_phase_obj_list_attrs("grid_v_mag")
masked_data_arr = np.array([masked_array(data, mask=disc_mask) for data, disc_mask in zip(data_arr, disc_mask_arr)])

plt.figure()
plt.contourf(test_obj.phase_00.grid_x, test_obj.phase_00.grid_z, masked_data_arr[0], levels=11, cmap='coolwarm')

masked_arr = masked_array(masked_data_arr[0], mask=disc_mask_arr[0])
plt.figure()
plt.contourf(test_obj.phase_00.grid_x, test_obj.phase_00.grid_z, masked_arr, levels=11, cmap='coolwarm')

#%% Masking in list comprehension test

import numpy as np
from numpy.ma import masked_array

# Example data and mask arrays
data_arr = [np.array([1, 2, 3]), np.array([4, 5, 6])]
disc_mask_arr = [np.array([False, True, False]), np.array([True, False, True])]

# Ensure the shapes match
for data, mask in zip(data_arr, disc_mask_arr):
    assert data.shape == mask.shape, "Data and mask shapes do not match"

# Create masked arrays using list comprehension
masked_data_arr = []

for data, mask in zip(data_arr, disc_mask_arr):
    print(data)
    print(mask)
    masked_data_arr.append(masked_array(data, mask=mask))
    
masked_data_arr = np.ma.array(masked_data_arr)

# Check the masked arrays
for masked_data, mask in zip(masked_data_arr, disc_mask_arr):
    print(masked_data)
    print(mask)

#%% testing get_attrs

# attribute test
print(test_obj.get_phase_obj_list_attrs("v_inf"))

# method test
test_obj.get_phase_obj_list_attrs("gen_data_intpr")
print(test_obj.get_phase_obj_list_attrs("grid_v_x_intpr", (-1,0)))

# %% Plotting

phase_id = "phase_00"
phase_obj = getattr(test_obj, phase_id)

v_inf = phase_obj.v_inf
p_inf = phase_obj.p_inf
rho_inf = phase_obj.rho_inf
v_app = phase_obj.v_inf - phase_obj.v_surge

disc_mask = phase_obj.disc_mask
x = phase_obj.grid_x
z = phase_obj.grid_z
v_x_raw = masked_array(phase_obj.grid_v_mag, mask=disc_mask)
p = masked_array(phase_obj.grid_p, mask=disc_mask)
v_x = phase_obj.unshdw_v_x
turb_intensity = masked_array(phase_obj.grid_turb_intensity, mask=disc_mask)
cp = (p - p_inf) / (0.5 * rho_inf * v_inf**2)
print("p_inf: ", p_inf, "dyn_pres: ", 0.5 * rho_inf * v_inf**2)

plt.figure()
v_x_raw_plot = plt.contourf(x,z,v_x_raw/v_inf,levels=np.linspace(0.2,1.1,11), norm=mpl.colors.TwoSlopeNorm(vcenter=1), cmap='coolwarm')
plt.gca().set_aspect('equal')
plt.colorbar(v_x_raw_plot, orientation='horizontal', label='v_x/v_inf')

plt.figure()
v_x_plot = plt.contourf(x,z,v_x/v_inf,levels=np.linspace(0.2,1.1,11), norm=mpl.colors.TwoSlopeNorm(vcenter=1), cmap='coolwarm')
plt.gca().set_aspect('equal')
plt.colorbar(v_x_plot, orientation='horizontal', label='v_x/v_inf')

plt.figure()
cp_plot = plt.contourf(x,z,cp,levels=11, norm=mpl.colors.TwoSlopeNorm(vcenter=0), cmap='coolwarm')
plt.gca().set_aspect('equal')
plt.colorbar(cp_plot, orientation='horizontal')

# plt.figure()
# turb_intensity_plot = plt.contourf(x,z,turb_intensity,levels=11, cmap='coolwarm')
# plt.gca().set_aspect('equal')
# plt.colorbar(turb_intensity_plot, orientation='horizontal', label='Turbulence intensity')
# plt.plot([0.5,0.5,1,1,0.5],[0.25,-0.25,-0.25,0.25,0.25],'k')

#%% Testing effective induced velocity
plt.plot(np.linspace(-1,1,101),phase_obj.get_eff_ind_intpr(v_ref=v_inf)(np.linspace(-1,1,101)))
plt.plot(np.linspace(-1,1,101),phase_obj.get_eff_ind_intpr(v_ref=v_app)(np.linspace(-1,1,101)))

# %% Paper Q1- Data

# Effective induction plots
plt.figure()
plt.plot(test_obj.phi_all_phases, test_obj.eff_ind_ground_all_phases, label='Ground reference')
plt.plot(test_obj.phi_all_phases, test_obj.unshdw_eff_ind_ground_all_phases, label='Unshdw ground reference',color="tab:blue",linestyle='--')
# plt.plot(test_obj.phi_all_phases, test_obj.eff_ind_disc_all_phases, label='Disc reference')
# plt.plot(test_obj.phi_all_phases, test_obj.unshdw_eff_ind_disc_all_phases, label='Unshdw disc reference',color="tab:orange",linestyle='--')
plt.axhline(*case_06_obj.eff_ind_ground_all_phases, color='g', linestyle='-', label='static v_inf=3')
plt.axhline(*case_06_obj.unshdw_eff_ind_ground_all_phases, color='g', linestyle='--', label='unshdw static v_inf=3')
plt.axhline(*case_07_obj.eff_ind_ground_all_phases, color='r', linestyle='-', label='static v_inf=2')
plt.axhline(*case_07_obj.unshdw_eff_ind_disc_all_phases, color='r', linestyle='--', label='unshdw static v_inf=2')
plt.xlabel('Phase angle (rad)')
plt.ylabel('Effective induction')
plt.legend()
plt.title(f'omega_star={test_obj.A_star_surge:.3g}, a_star={test_obj.omega_star_surge:.3g}, v_star={test_obj.omega_star_surge*test_obj.A_star_surge:.3g}') 


# %% Turbulence intensity plots
plt.figure()
plt.plot(test_obj.phi_all_phases, test_obj.avg_turb_intensity_all_phases, label='Average turbulence intensity')
plt.axhline(case_06_obj.avg_turb_intensity_all_phases.mean(), color='g', linestyle='-', label='static v_inf=3')
plt.axhline(case_07_obj.avg_turb_intensity_all_phases.mean(), color='r', linestyle='-', label='static v_inf=2')
plt.xlabel('Phase angle (rad)')
plt.ylabel('Turbulence intensity')
plt.legend()
plt.title(f'omega_star={test_obj.A_star_surge:.3g}, a_star={test_obj.omega_star_surge:.3g}, v_star={test_obj.omega_star_surge*test_obj.A_star_surge:.3g}')

# %%
