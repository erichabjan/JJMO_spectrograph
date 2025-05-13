import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter


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

def find_absorption_lines(wavelength, flux, d_range = 2, window_length=15, polyorder=1, end_buffer = 50):
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
    for i in range(1 + d_range + end_buffer, len(dflux) - d_range - end_buffer):
        # Check if the derivative has gone from negative to positive
        if all(dflux[i-1-d_range:i-1] < 0) and all(dflux[i:i+d_range] > 0):
            # Optional check: second derivative > 0 => local minimum
            # Otherwise, skip if you only want sign change
            if d2flux[i] > 0:
                minima_indices.append(i)
    
    # Convert indices to wavelength
    minima_waves = wavelength[minima_indices]
    
    return minima_indices, minima_waves