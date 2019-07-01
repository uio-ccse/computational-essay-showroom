---
redirect_from:
  - "essays/exercises/essayseeds/stormcloud"
interact_link: content/essays/exercises/essayseeds/StormCloud.ipynb
kernel_name: python3
has_widgets: false
title: 'Storm Cloud Template'
prev_page:
  url: /essays/exercises/essayseeds/MagneticBottle_Numpy
  title: 'Magnetic Bottle Template'
next_page:
  url: /essays/exercises/essayseeds/Stormcloud_Lightning
  title: 'Lightning Template'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Simulating a Storm Cloud in 3D

## Introduction

This notebook simulates a storm cloud, modeling it as a parallel-plate capacitor. It is based in large part on information from a report called "The Physics of Lightning" by Dwyer and Uman (*Dwyer, J. R., & Uman, M. A. (2014). The physics of lightning. Physics Reports, 534(4), 147–241. http://doi.org/10.1016/j.physrep.2013.09.004*. You can also find a short description of the basic concepts at http://hyperphysics.phy-astr.gsu.edu/hbase/electric/lightning.html#c1). The code calculates the electric field a certain distance away from the cloud, at a specified observation position. There are some suggestions for investigation questions and the end, but you are welcome to play around with the parameters in the code, add or subtract different pieces, and see how it behaves for yourself as you decide what you want to use it to investigate.

## Simulation of the Storm Cloud

First, we import our various libraries: sympy, numpy, and matplotlib



{:.input_area}
```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
```


Now, we will define some constants: the dimensions of the cloud, altitude of the cloud, charge of the positive and negative part of the cloud, and the observation position (where we are observing the electric field from the cloud



{:.input_area}
```python
startx = -2500 #Define where the parts of the clouds should start/stop in the x direction
endx = 2500

startz = -2500 #Define where the cloud should start/stop in the z direction
endz = 2500

negheight = 6000 #set the negative cloud at a height (in the y direction) of 6000m
posheight = 8000 #set the positive cloud at a height of 8000m

Q = -15 #Q is the total charge on the bottom (negative) part of the cloud
Q2 = 15 #Q2 is the total charge of the top (positive) part of the cloud

k = 9e9 #Coulomb's constant

obspos = np.array([0,0,0]) #the observation position (start at 0,0,0)
```


Now, we will define the chunks we break the clouds (and net charge) into. First, we'll define how many "chunks" we'll break the cloud into in the x and z direction, the size of each of those chunks (based on the overall size of the cloud) and the charge of each of those chunks 



{:.input_area}
```python
nx = 100 #Define how many chunks to split the cloud in the x direction (define x size of the cloud grid)
nz = 100 #Define how many chunks to split the cloud in the x direction (define x size of the cloud grid)

stepx = (endx - startx)/nx #Define the spacing between each chunk in the x direction
stepz = (endz - startz)/nz #Define the spacing between each chunk in the z direction

dQ = Q/(nx*nz) #Charge of each chunk of the the negative part of the cloud
dQ2 = Q2/(nx*nz) #Charge of each chunk of the positive part of the cloud
```


Finally, we will define the e-field variable, initialize it to 0, and calculate the net e-field by iterating over each of these chunks and adding each of their contributions to the net e-field.

(Note that "np.linalg.norm" essentially takes the magnitude of a vector or array, so we use that in calculating the e-field)



{:.input_area}
```python
efield = 0

for i in range(0,nx): #iterate over the x dimension of the cloud
    xloc = startx + i*stepx
    for j in range(0,nz): #iterate over the z dimension of the cloud
        zloc = startz + j*stepz
    
        negfield = k*dQ/(np.linalg.norm(obspos-np.array([xloc,negheight,zloc])))**2
        posfield = k*dQ2/(np.linalg.norm(obspos-np.array([xloc,posheight,zloc])))**2

        efield = efield + negfield + posfield
        
print("The e-field at the observation position is", efield, "Newtons per coulomb")
```


{:.output .output_stream}
```
The e-field at the observation position is -1391.7881530213551 Newtons per coulomb

```

With 100 by 100 chunks, we get an e-field of -1391.788 N/C. From other tests this is about .002% different from the value with 1000 by 1000 chunks, so it seems that 100 chunks in x and z will work just as well as 1000.

## Additional questions you might investigate

1. How likely is it for lightning to strike a particular spot, knowing that the electric field breakdown of air is about 3e6 V/M? How far away from the cloud would you have to be to be safe from lightning?
    * What if the cloud polarizes the ground, or an object near the ground?
2. Actual clouds have a certain thickness. How does this calculation change if the clouds are not thin sheets, but are 3D instead?
3. It turns out that in reality, there are multiple layers of + and - charge, each with somewhat different charge densities (see *Marshall, T. C., & Stolzenburg, M. (1998). Estimates of cloud charge densities in thunderstorms. Journal of Geophysical Research, 103(D16), 19769–19775.*) What happens if there are extra layers of positive or negative charge in these clouds?
4. What if the cloud is larger or smaller? Higher up or closer to the ground?

*(Note that these are just meant to be suggestions—feel free to investigate any question you find interesting!)*

For more information on clouds and lightning (including approximate numbers for many of the physical characteristics of storm clouds) see *Dwyer, J. R., & Uman, M. A. (2014). The physics of lightning. Physics Reports, 534(4), 147–241. http://doi.org/10.1016/j.physrep.2013.09.004*
