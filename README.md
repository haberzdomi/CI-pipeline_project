## Overview

This project determines the magnetic field of the tokamak of the ASDEX Upgrade. The coil geometry and the currents flowing through the coils are defined in co_asd.dd and cur_asd.dd, respectively. In a first step biotsavart_asdex.py calculates the magnetic field for a discretized grid which is defined in biosavart.inp. Then plot_modes.py reads in the output of this calculation (field.dat) and determines the first n modes of the magnetic field by fourier transformation. Afterwards it creates n colorplots of the modes of the magnetic field. In addition plot_coils.py can be used to draw the coil geometry.

## Executeables

###### biotsavart_asdex.py

Take the input parameters of 'biotsavart.inp' which define a 3D-grid for which the magnetic field components are calculated using the Biot-Savart law. The output (covariant components of the magnetic field) is written to 'field.dat'.

###### plot_modes.py

Read the output of the biotsavart_asdex calculation ('field,dat') and its input parameters ('biotsavart.inp'). Fourier transformation is used to calculate the the first n modes of R- and Z-component of the magnetic field. The phi component is calculated such that the magnetic field gets divergence-free. Precisely the decadic logarithm of the square of the norm of the total magnetic field is caluculated for each of the modes on a grid which has double the resolution of the 3D-grid defined for biotsavart_asdex.py. For each modem one colorplot is created.

###### plot_coils.py

Create a 3D plot of the coils (see coils_geometry.png). The geometry of the coils is read from 'co_asd.dd'.

## Structure

This project is organized as shown in call_graph.pdf which provides in addition to the call graph a short description for all files and functions as well as information about global variables.
