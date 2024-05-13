# How feedback influences Redshift-Space Distortions

This directory contains the relevant source code of my project on Redshift-Space Distortions at Leiden University.
Data used is from the FLAMINGO simulations, which are cosmological (computer-)simulations of large-scale structure of the Universe at scales within a 1 Gpc box. 

The project is about extracting and analysing data on the galaxies inside the simulated Universe, more specifically the data about the stars inside the galaxies. 
We test how different feedback mechanisms like stellar mass, redshift and AGN feedback influence the redshift-space distortion (RSD) effects. These RSD effects manifest in two ways.
The Kaiser effect occurs on larger scales due to the infalling motion of galaxies towards the gravitational center of a galaxy cluster, and the Fingers-of-God effect happens
on smaller scales as a result of the random peculiar velocities of galaxies inside clusters. 

## Code order
The python files should be read as follows:

- properties.py : We begin by reading out data from the FLAMINGO simulations and saving in a binary format 
- lookup_table.py : Creates a look-up table to use in the define_galaxies.py routine, more specifically to significantly speed up the masking of halo groups
- define_galaxies.py : Construct a galaxy catalogue using the particle properties, including galaxy position, velocity, velocity dispersion and stellar mass
- FFTPower_custom.py : Use nbodykit's FFTPower to create a custom function that projects the FFT data-cube to a custom basis in two dimensions
-

### Credits

- Leiden University: https://www.universiteitleiden.nl/
- FLAMINGO simulations: https://flamingo.strw.leidenuniv.nl/
- nbodykit: https://nbodykit.readthedocs.io/en/latest/

### Code by Jim van Veenhuyzen, MSc student Astronomy Research at Leiden University, 2024
