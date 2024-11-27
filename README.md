## Overview

This project determines the magnetic field for a given coil geometry, defined in coil_file and the currents flowing through the coils defined in current_file, respectively. The calculated magnetic field is then saved together with some input information in field_file. In a first step make_field_file_from_coils of biotsavart.py calculates the magnetic field for the given discretized grid. Then plot_modes.py reads the output of this calculation (field_file) and determines the first n modes of the magnetic field by fourier transformation. Also it ensures that the field is divergency-free in every point using its vector potential. Afterwards it creates n colorplots of the modes of the magnetic field. In addition plot_coils.py can be used to draw the coil geometry.

## Executeables

###### biotsavart_asdex.py

Take the input parameters of 'grid_file', 'coil_file' and 'current_file' to calculate the magnetic field components using the Biot-Savart law. The output (magnetic field components for each point on the 3D grid) is written to 'field_file'.

###### plot_modes.py

Read the output of the magnetic field calculation ('field,dat') and use Fourier transformation to calculate the the first n modes of the R- and Z-component of the magnetic field. For this modes the vector potential is calculated by B=rot(A) and then a spline approximation is done to evaluate all points in space. The phi component of the vector potential is calculated such that the magnetic field gets divergence-free. The decadic logarithm of the square of the norm of the total magnetic field is caluculated for each of the modes on a grid which has double the resolution of the 3D-grid of the magnetic field calculation. For each modem one colorplot is created.

###### plot_coils.py

Create a 3D plot of the coils (see coils_geometry.png). The geometry of the coils is read from 'coil_file'.

## Structure

This project is organized as shown in call_graph.pdf which provides in addition to the call graph a short description for all files and functions as well as information about global variables. (Outdated)
