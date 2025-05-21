import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter

### Define Absorption and Telluric transtions

balmer_lines = np.array([
    6562.8,  # Hα
    4861.3,  # Hβ
    4340.5,  # Hγ
    4101.7,  # Hδ
    3970.1,  # Hε
    3889.1,  # Hζ
    3835.4   # Hη
])

metal_lines = np.array([
    3933.7,  # Ca II K
    3968.5,  # Ca II H (overlaps with Hε)
    4481.2,  # Mg II (triplet)
    4383.6,  # Fe I
    4923.9,  # Fe II
    5018.4,  # Fe II
    5169.0,  # Fe II
    4552.6,  # Si III
    4077.7   # Sr II
])

telluric_lines = np.array([
    6277.0,  # O2 "A" band
    6867.0,  # H2O band (B band region)
    7186.0,  # H2O
    7594.0,  # O2 "B" band
    7630.0   # O2 (deepest in B band)
])


### Function to import data as a single numpy array

def import_to_fits(path_array):
    
    hdulist = []
    
    wave_arr = np.array([np.genfromtxt(file + '.txt', delimiter='\t', usecols=(1), invalid_raise=False) for file in path_array])
    
    for i in range(len(wave_arr)):
        
        wave = wave_arr[i]
        spec = np.flip(np.nansum(fits.open(path_array[i] + '.fit')[0].data, axis=0))
        spec_header = fits.open(path_array[i] + '.fit')[0].header
    
        ### Input wave and spectrum should be numpy arrays of the same length 
        ### Input header should be a FITS header 
    
        ### The information that I am taking from 'spec_header' is minimal at the moment and can be added upon
    
        crval1, crpix1, crdelt1 = np.min(wave), 1, np.mean(abs(np.diff(wave)))

        primary_hdu = fits.PrimaryHDU(spec)

        # Define header information
        primary_hdu.header['CRVAL1'] = (crval1, 'Coordinate Reference Value in Angstroms')  # Example wavelength reference value
        primary_hdu.header['CRPIX1'] = (crpix1, 'Coordinate Reference Pixel')      # Reference pixel
        primary_hdu.header['CDELT1'] = (crdelt1, 'Coordinate Incremant in Angstroms')    # Wavelength increment
        primary_hdu.header['CTYPE1'] = ('WAVE', 'Coordinate Type')  # Coordinate type (arbitrary label)
        primary_hdu.header['BUNIT'] = ('Angstrom', 'Unit')  # Unit of measurement for the data

        if 'DATE' in spec_header:
            primary_hdu.header['DATE'] = (spec_header['DATE'], 'Observation Date (UT)')
        
        if 'TIME-OBS' in spec_header:
            primary_hdu.header['TIME-OBS'] = (spec_header['TIME-OBS'], 'Observation Time (UT)')
    
        if 'INSTRUME' in spec_header:
            primary_hdu.header['INSTRUME'] = (spec_header['INSTRUME'], 'Instrument Name')
        
        if 'EXPTIME' in spec_header:
            primary_hdu.header['EXPTIME'] = (np.float64(spec_header['EXPTIME']), 'Exposture time in Seconds')

        # Create an HDU list and write to a FITS file
        hdulist.append(fits.HDUList([primary_hdu])[0])
    
    return hdulist

def find_absorption_lines(wavelength, flux, d_range = 5, window_length=2, polyorder=1, end_buffer = 10):
    """
    Find approximate local minima (absorption lines) via a derivative-based approach.
    
    Parameters
    ----------
    wavelength : array-like
        Wavelength values in ascending order (e.g., Angstrom).
    flux : array-like
        Measured intensity values corresponding to each wavelength.
    window_length : int, optional
        Window size for Savitzky-Golay smoothing.
    polyorder : int, optional
        Polynomial order for Savitzky-Golay smoothing.
        
    Returns
    -------
    minima_indices : list of int
        Indices of flux array where local minima are found.
    minima_waves : list of float
        Wavelength values corresponding to those minima.
    """
    
    # 1. (Optional) Smooth the flux to reduce noise
    #    Adjust window_length & polyorder based on your data's resolution and SNR.
    flux_smooth = savgol_filter(flux, window_length=window_length, polyorder=polyorder)
    
    # 2. Compute the first derivative with respect to wavelength.
    #    np.gradient() automatically approximates derivative from discrete data
    dflux = np.gradient(flux_smooth, wavelength)
    
    # 3. (Optional) compute the second derivative to confirm concavity
    d2flux = np.gradient(dflux, wavelength)
    
    minima_indices = []
    
    # 4. Identify zero-crossings of dFlux (neg -> pos) and concavity (2nd deriv > 0)
    for i in range(d_range + end_buffer, len(dflux) - d_range - end_buffer):
        # Check if the derivative has gone from negative to positive
        if all(dflux[i-d_range:i] < 0) and all(dflux[1+i:i+d_range+1] > 0):
            
            # Optional check: second derivative > 0 => local minimum
            # Otherwise, skip if you only want sign change
            if d2flux[i] > 0:
                minima_indices.append(i)
    
    # Convert indices to wavelength
    minima_waves = wavelength[minima_indices]
    
    return minima_indices, minima_waves

def wavelength_velocity(fits_in):
    
    vel_specs = []
    wave_specs = []
    wave_list = []
    
    c = 3 * 10**6
    
    for k in range(len(fits_in)):

        ### Make lists of each wavelength axis
        spec_len = fits_in[k].header['NAXIS1']
        lower = fits_in[k].header['CRVAL1']
        upper = fits_in[k].header['CRVAL1'] + fits_in[k].header['NAXIS1'] * fits_in[k].header['CDELT1']
        wave_axis = np.linspace(lower, upper, spec_len)
        wave_list.append(wave_axis)
    
        ### Define what absorption/telluric lines are in each spectra
        balmer_k = balmer_lines[(balmer_lines > lower) & (balmer_lines < upper)]
        metal_k = metal_lines[(metal_lines > lower) & (metal_lines < upper)]
        telluric_k = telluric_lines[(telluric_lines > lower) & (telluric_lines < upper)]
    
        ### find which spectra have both telluric and absorption lines 
        if (len(balmer_k) > 0 or len(metal_k) > 0) and len(telluric_k) > 0:
            vel_specs.append(k)
        
        else: 
            wave_specs.append(k)
    
    ### Arrays for final velocity of object and offsets of each spectrum        
    velocities = np.zeros(len(fits_in))
    offsets = np.zeros(len(fits_in))
    
    for k in vel_specs: 
        
        ### Derivative code to find minima
        ab_ind, absorption_lines = find_absorption_lines(wave_list[k], fits_in[k].data)
        ### Redifine lines in spectra
        balmer_k = balmer_lines[(balmer_lines > np.min(wave_list[k])) & (balmer_lines < np.max(wave_list[k]))]    
        metal_k = metal_lines[(metal_lines > np.min(wave_list[k])) & (metal_lines < np.max(wave_list[k]))]
        telluric_k = telluric_lines[(telluric_lines > np.min(wave_list[k])) & (telluric_lines < np.max(wave_list[k]))]
        ### list of offsets with respect to telluric and absorption lines
        tell_offset = []
        
        for j in range(len(telluric_k)):

            tell_bool = np.min(np.abs(absorption_lines - telluric_k[j])) == np.abs(absorption_lines - telluric_k[j])
            
            if np.abs(absorption_lines[tell_bool] - telluric_k[j]) < 50:
                tell_offset.append(absorption_lines[tell_bool] - telluric_k[j])
            else:
                continue
        
        ### Add the the median offset to the array
        offsets[k] = np.median(tell_offset)
        ### Combine absorption line lists
        lines = np.concatenate((balmer_k, metal_k))
        
        for j in range(len(lines)):
            
            ### Find minimum wavelangth offset
            ab_bool = np.min(np.abs(absorption_lines - lines[j])) == np.abs(absorption_lines - lines[j])
            if np.abs(absorption_lines[ab_bool] - lines[j]) < 50:
                ### Find velocity for reasonable (offsets less than 50 angstroms) absorption lines 
                ab_offset = absorption_lines[ab_bool] - lines[j]
                z = np.abs(ab_offset - offsets[k]) / absorption_lines[ab_bool]
                velocities[k] = z[0] * c
            else:
                continue
    
    return offsets, velocities