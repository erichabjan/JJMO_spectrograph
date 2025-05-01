import numpy as np
from astropy.io import fits



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