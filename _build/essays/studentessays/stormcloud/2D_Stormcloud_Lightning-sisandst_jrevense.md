---
redirect_from:
  - "essays/studentessays/stormcloud/2d-stormcloud-lightning-sisandst-jrevense"
interact_link: content/essays/studentessays/stormcloud/2D_Stormcloud_Lightning-sisandst_jrevense.ipynb
kernel_name: python3
has_widgets: false
title: 'What is the best material for a lightning rod?'
prev_page:
  url: /index
  title: 'Student Essays'
next_page:
  url: /essays/studentessays/LHC/LHC_CompEssay
  title: 'What are the effects of relativity on particles in the LHC?'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Computational Essay Project FYS1120

## Contact info:

#### Synnøve Isabel Wallentin Sandstrøm

E-mail: sisandst@student.matnat.uio.no

#### Jarl-Robin Bjerkgård Evensen

E-mail: jrevense@student.matnat.uio.no

# Introduction: 2D Stormcloud, with lightning and lightning rod

In this notebook, we will model the path of a bolt of lightning when there is a lightning rod present. This simulation will create a grid of points and then solve Poisson's equation on that grid in order to find the electric potential throughout the 2D space. Based on that electric potential, it will predict where the lightning strike begins, then repeat the process to see how it advances.

After we've modelled the lightning strike we will see how a lightning rod setup will handle being struck by lightning, more specifically how it will handle the amount of energy in the strike and then evaluate whether the setups seem safe.

To calculate this we make simplified models of the rods and wires, calculate the resistances for different materials and gauges of wire, calculate for a range of currents in the strikes and finally compare the temperature rise in the system to the melting point of the PVC insulation around the wires.

First, we import the essential tools: numpy, matplotlib, pandas and numba to speed up calculations, and handle data.



{:.input_area}
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from numba import jit
import pandas as pd
```


Next, we define a function that will solve Poisson's equation on a rectangular grid, using a method called the "method of relaxation." This method relies on the fact that, as long as there are no charges in a region, the potential in that region will only change gradually. So, given a set of boundary conditions (places where the potential is specified or well defined) we can find the values of electric potential at certain points by averaging the potential of neighboring points. By doing this over and over, we eventually get a stable solution.

In the following function, the boundary conditions are given by the array b. In the points where b is defined (has a value), the boundary conditions are assumed to be this value. Where b is 'nan' (not defined—that is, we have yet to find it) we use the method of relaxation to find the value.



{:.input_area}
```python
@jit(cache=True)
def solvepoisson(b,nrep):
    # b = boundary conditions
    # nrep = number of iterations

    z = np.copy(b)     # z = electric potential field
    j = np.where(np.isnan(b)) #checks for where the points have no value, assigns them the value 0
    z[j] = 0.0
    
    znew = np.copy(z)
    Lx = np.size(b,0) # determine the x range of the point grid
    Ly = np.size(b,1) # determine the y range of the point grid
    
    for n in range(nrep): 
        for ix in range(Lx):
            for iy in range(Ly):
                ncount = 0.0 
                pot = 0.0
                if (np.isnan(b[ix,iy])): # check for cases in which the value is unspecified in the original grid
                    # Now, add up the potentials of all the the points around it
                    if (ix>0): 
                        ncount = ncount + 1.0
                        pot = pot + z[ix-1,iy]
                    if (ix<Lx-1):
                        ncount = ncount + 1.0
                        pot = pot + z[ix+1,iy]
                    if (iy>0):
                        ncount = ncount + 1.0
                        pot = pot + z[ix,iy-1]
                    if (iy<Ly-1):
                        ncount = ncount + 1.0
                        pot = pot + z[ix,iy+1]
                    znew[ix,iy] = pot/ncount # Divide by the number of contributing 
                                             # surrounding points to find average potential
                else:
                    znew[ix,iy]=z[ix,iy] # If the value is specified, keep it
        tmp_z = znew # Swapping the field used for the calucaltions with the field from the previous iteration
        znew = z     # (to prepare for the next iteration)
        z = tmp_z     
    return z 
```


Now, we will use the poisson solver to simulate lightning. First, we will set up the boundary conditions, creating a grid of 50 by 50 points, setting all values to 'nan' (meaning they need to be solved for) except at the top (the cloud) where the potential is specified to be 2, and at the bottom (the ground) where it is specified to be zero.
<br>

We have to calculate the surface potential of the conductor first, this might seem daunting at first, we did a lot of back and forth to figure this out to be honest, untill we realised that we only need to deal with a constant potential. Since the lightning rod is connected to ground all charge within it and the conductor connecting it to ground will be evenly distributed across the surface of the rod, this means that the entire system will have a constant potential. We decided to set the potential to be $\frac{1}{10}$ of the top potential.



{:.input_area}
```python
def set_conditions(Lx = 50, Ly = 50, L = 2, top_pot = 2.0, bot_pot = 0.0):

    # First, we set up the boundary conditions
    # and the necessary variables for our lightning rod

    rod_pot = top_pot/10.0
    rod_height = L/Ly

    z = np.zeros((Lx,Ly),float)
    b = np.copy(z)
    c = np.copy(z)
    b[:] = np.float('nan')

    # Set the potential at the top of the grid to top_pot
    b[:,0] = top_pot
    # Set the potential at the bottom of the grid to bot_pot
    b[:,Ly-1] = bot_pot

    return z, b, c, Lx, Ly, rod_pot, rod_height
```


Now, to add in the lightning. To simulate the path of the lightning strike we will work backwards, starting with some charge on the ground and seeing where it moves to in order to get up to the cloud (this is simulating so-called "ground to cloud" lightning). This charge, because it is negative, prefers to stay at low potential, in this case V = 0. So, to find the path of the lightning, we will proceed as follows:

1. Use the poisson solver to find the potential across the entire space, ground to cloud, starting with the ground
2. Find where the charge is most likely to move to, based on the potential values multiplied by a random factor
3. Set that location's potential equal to 0
4. Update the neighboring positions to the lightning's path, making them possible locations for the lightning's next move
5. We also set the potential where the lightning rod is placed, constant for each iteration of the loop.

In practice, this means we will be working with three arrays: The first holds the boundary values and the lightning's path (we call that one **b**; it was already defined, but will be updated based on the lightning's path). The second holds the possible places the lightning can move, stored as 'nan' values (we call that one **zeroneighbor**). The last holds the probabilistic values that lighting will move to particular positions (we call that one **sprob**).



{:.input_area}
```python
# Create a function to calculate the bolt

def calc_bolt(z, b, c, Lx, Ly, rod_pot, rod_height):
    # Create a copy of the boundary conditions matrix which will be used to check
    #for possible locations for the lightning's path
    zeroneighbor = np.copy(z)
    zeroneighbor[:] = 0.0 #set all values in it equal to 0
    #set the values next to the ground equal to 'nan'. This is where the lightning can start
    zeroneighbor[:,Ly-2] = np.float('nan')


    nrep = 3000 # Number of jacobi steps
    eta = 1.0 #A factor that will be used in probability calculation
    ymin = Ly-1 #The y value where we will stop (just above the ground)
    ns = 0
    i = 0

    while (ymin>0):
        # First find potential on the entire grid, based on the original boundary conditions
        s = solvepoisson(b,nrep)

        # Probability that lightning will move to a new position may depend on potential to power eta
        sprob = s**eta
        # We also multiply by a random number, uniform between 0 and 1, to introduce some randomness
        # And we multiply with isnan(zeroneighbor) to ensure only 'nan' points can be chosen
        sprob = sprob*np.random.uniform(0,1,(Lx,Ly))*np.isnan(zeroneighbor)

        #now, find the point with max probability
        [ix,iy] = np.unravel_index(np.argmax(sprob,axis=None),sprob.shape)

        # Update the boundary condition array to set the potential where the lightning is to 0
        b[ix,iy] = 0.0

        # Update neighbor positions of the lightning path to 'nan' 
        #(making them possible choices for the next iteration)
        if (ix>0):
            zeroneighbor[ix-1,iy]=np.float('nan')
        if (ix<Lx-1):
            zeroneighbor[ix+1,iy]=np.float('nan')
        if (iy>0):
            zeroneighbor[ix,iy-1]=np.float('nan')
        if (iy<Ly-1):
            zeroneighbor[ix,iy+1]=np.float('nan')

        x_pos = int(len(b[0])/2)
        y_pos = len(b[1])
        b[x_pos:x_pos+1,int(y_pos*(1-rod_height)):y_pos] = rod_pot  
        zeroneighbor[x_pos:x_pos+1,int(y_pos*(1-rod_height)):y_pos] = rod_pot
        # sets the potential around the lightning rod in both b and zeroneighbour
        # this is to make sure that the lightning goes past the rod.


        ns = ns + 1
        c[ix,iy] = ns #create an array of the lightning's path, scaled by the number of loops
        if (iy<ymin): #iterate to the next set of y-values
            ymin = iy

        i+=1
        if i > 10000:
            break
    return c, s, sprob
```




{:.input_area}
```python
plt.figure(1,figsize=(20,15))
z, b, c, Lx, Ly, rod_pot, rod_height = set_conditions(Lx = 100, Ly = 100)
c1,s1,sprob1 = calc_bolt(z,b,c,Lx,Ly, rod_pot, rod_height)
z, b, c, Lx, Ly, rod_pot, rod_height = set_conditions(Lx = 100, Ly = 100)
c2,s2,sprob2 = calc_bolt(z,b,c,Lx,Ly, rod_pot, rod_height)
z, b, c, Lx, Ly, rod_pot, rod_height = set_conditions(Lx = 100, Ly = 100)
c3,s3,sprob3 = calc_bolt(z,b,c,Lx,Ly, rod_pot, rod_height)
z, b, c, Lx, Ly, rod_pot, rod_height = set_conditions(Lx = 100, Ly = 100)
c4,s4,sprob4 = calc_bolt(z,b,c,Lx,Ly, rod_pot, rod_height)

plt.subplot(4,2,1)
plt.imshow(c1.T, cmap="Purples_r") #create a plot of the lightning's path
plt.colorbar()

plt.subplot(4,2,2)
plt.imshow(s1.T, cmap="Purples_r") #create a plot of the final potential
plt.colorbar()

plt.subplot(4,2,3)
plt.imshow(c2.T, cmap="Purples_r") #create a plot of the lightning's path
plt.colorbar()

plt.subplot(4,2,4)
plt.imshow(s2.T, cmap="Purples_r") #create a plot of the final potential
plt.colorbar()

plt.subplot(4,2,5)
plt.imshow(c3.T, cmap="Purples_r") #create a plot of the lightning's path
plt.colorbar()

plt.subplot(4,2,6)
plt.imshow(s3.T, cmap="Purples_r") #create a plot of the final potential
plt.colorbar()

plt.subplot(4,2,7)
plt.imshow(c4.T, cmap="Purples_r") #create a plot of the lightning's path
plt.colorbar()

plt.subplot(4,2,8)
plt.imshow(s4.T, cmap="Purples_r") #create a plot of the final potential
plt.colorbar()

plt.tight_layout()
plt.show()
```



{:.output .output_png}
![png](../../../../images/essays/studentessays/stormcloud/2D_Stormcloud_Lightning-sisandst_jrevense_10_0.png)



After running these simulations and plots dozens of times, we see that the majority of the lightning strikes will strike the lightning rod. Occasionally we get a lightning strike that does not strike the rod, and in these cases we noted that the strike struck quite a distance from the rod. This is due to the rod making a 45 degree "cone of protection" under it, as shown in the figure below.

This means that if we want to protect large structures, we have to set up several rods to protect it well enough. (see figure below with overhead ground wire)

<img src="https://cdn.britannica.com/s:700x450/31/24031-004-8D61DAC0.jpg" alt="45 degree cone of protection image" title="Image from Britannica.com" />

__[Source](https://www.britannica.com/technology/lightning-rod)__



We'll now look at these types of metals with conductivity $\sigma$, specific heat $H_{C}$ and density $\rho$.

Gold: 
<br>
$\sigma = 4.1 \cdot 10^{7}\ S/m$
<br>
$H_{C} = 125.604\ J / (kg K)$
<br>
$\rho = 19300\ \text{kg/m}^{3}$

Copper: 
<br>
$\sigma = 5.96 \cdot 10^{7}\ S/m$
<br>
$H_{C} = 376.812\ J / (kg K) $
<br>
$\rho = 8960\ \text{kg/m}^{3}$

Aliminium: 
<br>
$\sigma = 3.77 \cdot 10^{7}\ S/m$
<br>
$H_{C} = 921.096\ J / (kg K)$
<br>
$\rho = 2600\ \text{kg/m}^{3}$

Silver: 
<br>
$\sigma = 6.3 \cdot 10^{7}\ S/m$
<br>
$ H_{C} = 238.6476\ J / (kg K) $
<br>
$\rho = 10500\ \text{kg/m}^{3}$

Iron: 
<br>
$\sigma = 1 \cdot 10^{7}\ S/m$
<br>
$ H_{C} = 460.548\ J / (kg K)$
<br>
$\rho = 7870\ \text{kg/m}^{3}$

We can calculate the resistance in a conductor with the formula: $R = \frac{L}{A\sigma}$ where $L$ is the length of the conductor, $A$ is the cross sectional area and $\sigma$ is the conductivity of the conductor.



We want to calculate the energy going through a conductor(the lightning rod and wire in this case), to do this we can use the formula $T = \frac{E}{H_{C}\ m}$, where E is the energy being passed through the system, $H_{C}$ is the specific heat of the conductor and $m$ is the mass of the conductor.
<br>

We know that $E = P t$ where $P = I^2R$, $t$ is the time the current is passed, inserting all this gives us:
<br>

$$ T = \frac{I^{2}Rt}{H_{C}m}  $$
<br>


We now have to find some baseline currents for our lightning strikes, through our research we found that lightning strikes can have very varying currents, from as low as 3kA to as high as 200kA. We decided to test the entire range of currents for a couple of different lengths of materials

__[Source](https://hypertextbook.com/facts/1997/BrookeHaramija.shtml)__

Lightning rods are connected to ground by a lot of different means, some are connected to the plumbing of the building, some are connected to structural metal frames and some are simply connected to ground by a wire from the rod. 

Estimating the amount of metal in a house's plumbing or structural frames is not that interesting in this case, since the amount of metal will easily be able to dissipate the energy from a lightning strike. We decided to look at some standard gauge wire, with PVC insulation, of different lengths. 

We deem a setup safe if the temperature rise in the wire is below the melting point of the PVC insulation, which is 
105 degrees celcius, if the conductor material heats up more than this we would deem it unsafe for use in this application.

__[Source](https://www.awcwire.com/insulation-materials)__



{:.input_area}
```python
# Here we define a dictionary with the values of the different properties

names = ["Gold", "Copper", "Aluminium", "Silver", "Iron"] # Conductor names
sigma = [(4.1* 10**7), (5.96* 10**7), (3.77* 10**7), (6.3* 10**7), (1* 10**7)] # S/m
spec_heat = [125.604, 376.812, 921.096, 238.6476, 460.548] # J / (kg K)
rho = [19300, 8960, 2600, 10500, 7870] # kg/m^3

materials = {}               # setup an empty dictionary to populate
for i in range(len(names)):  # populate the dictionary with the given values
    values = {}
    values["sigma"] = sigma[i]
    values["spec_heat"] = spec_heat[i]
    values["rho"] = rho[i]
    materials[names[i]] = values
```




{:.input_area}
```python
# Define a couple of functions for calculating Ohm's law and the temperature increase
# in a wire for a given wire

def resistance(L, A, sigma):
    """
    Calculates the resistance of a given length of wire
    """
    return (L / (A*sigma))

def temp(R, I, t, Hc, Vol, rho):
    """
    Calculate the increase in temperature based 
    on the energy passed in the conductor
    """
    return ((R * (I**2) * t) / (Hc * Vol * rho))
```


We found differing regulations about roof heights in Norway, from our research we decided that 4.5 meters and 8.0 meters were sensible values for single storey and two storey buildings respectively. 

In our calculations we choose to assume that the cable connecting the rod to ground goes straight down to ground from the roof, to account for the length of wire that needs to be in the ground for a proper grounding to take place we assumed 0.5 metres would be sufficient.



{:.input_area}
```python
# Setup the one storey and two storey house parameters.

one_storey_rod = 1.0    #m
two_storey_rod = 1.5    #m
rod_gauge = (0.01**2) * np.pi # m^2

one_storey_wire = 5.0   #m
two_storey_wire = 8.5   #m

wire_gauge = np.array([120, 95, 70, 55, 50, 35, 25, 16, 10, 6]) * (10**-6) #m^2   
# These values are AWG 4/0, 3/0, 2/0, 1/0, 1, 2, 4, 6, 8 and 10 in m^2

for conductor in materials:
    R_rod_os = resistance(one_storey_rod, rod_gauge, materials[conductor]["sigma"])# Calc resistance of one storey rod
    R_rod_ts = resistance(two_storey_rod, rod_gauge, materials[conductor]["sigma"])# Calc resistance of two storey rod
    
    R_os = resistance(one_storey_wire, wire_gauge, materials[conductor]["sigma"])# Calc resistance of one storey wire
    R_ts = resistance(two_storey_wire, wire_gauge, materials[conductor]["sigma"])# Calc resistance of one storey wire
    
    resistances = {}
    resistances["os"] = (R_rod_os + R_os) # sums the rod and wire resistances
    resistances["ts"] = (R_rod_ts + R_ts) # sums the rod and wire resistances
    
    materials[conductor]["resistances"] = resistances # adds the resistances into our main dictionary
```




{:.input_area}
```python
# Here we make a couple of nice looking tables using pandas

gauges = ["4/0","3/0","2/0","1/0","1","2","4","6","8","10"]  # wire gauges
mat_list = [r"$Gold\ R(\Omega)$",r"$Copper\ R(\Omega)$",r"$Aluminium\ R(\Omega)$",
            r"$Silver\ R(\Omega)$", r"$Iron\ R(\Omega)$"]    # names with latex syntax for the plots and tables

resistances_os = [] 
resistances_ts = []

for conductor in materials:
    resistances_os.append(materials[conductor]["resistances"]["os"])
    resistances_ts.append(materials[conductor]["resistances"]["ts"])
    

res_table_os = pd.DataFrame(resistances_os, 
                            columns=gauges, 
                            index=mat_list)

res_table_ts = pd.DataFrame(resistances_ts, 
                            columns=gauges, 
                            index=mat_list)
```




{:.input_area}
```python
def highlight_min(s):
    '''
    highlight the minimum in a Series.
    '''
    is_min = s == s.min()
    return ['background-color: lightblue' if v else '' for v in is_min]

def highlight_max(s):
    '''
    highlight the max in a Series.
    '''
    is_max = s == s.max()
    return ['background-color: red' if v else '' for v in is_max]

def color_val(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for values over 1
    strings, black otherwise.
    """
    if val > 1:
        color = 'pink'
    elif 1 >= val > 0.75:
        color = 'darkorange'
    elif 0.75 >= val > 0.5:
        color = 'orange'
    elif 0.25 >= val > 0:
        color = 'green'
    else:
        color = 'black'
    return 'color: %s' % color
```


Here we can see a couple of tables which lists the resistances of our different situations, the first is the resistances of the different wire gauges, and materials, for one storey buildings, the second is for two storey buildings.



{:.input_area}
```python
res_table_os.style.apply(highlight_min).apply(highlight_max)
```





<div markdown="0" class="output output_html">
<style  type="text/css" >
    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col0 {
            background-color:  lightblue;
            : ;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col1 {
            background-color:  lightblue;
            : ;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col2 {
            background-color:  lightblue;
            : ;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col3 {
            background-color:  lightblue;
            : ;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col4 {
            background-color:  lightblue;
            : ;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col5 {
            background-color:  lightblue;
            : ;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col6 {
            background-color:  lightblue;
            : ;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col7 {
            background-color:  lightblue;
            : ;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col8 {
            background-color:  lightblue;
            : ;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col9 {
            background-color:  lightblue;
            : ;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col0 {
            : ;
            background-color:  red;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col1 {
            : ;
            background-color:  red;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col2 {
            : ;
            background-color:  red;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col3 {
            : ;
            background-color:  red;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col4 {
            : ;
            background-color:  red;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col5 {
            : ;
            background-color:  red;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col6 {
            : ;
            background-color:  red;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col7 {
            : ;
            background-color:  red;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col8 {
            : ;
            background-color:  red;
        }    #T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col9 {
            : ;
            background-color:  red;
        }</style>  
<table id="T_acdc8edc_1a60_11e9_b616_0a580ae941de" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >4/0</th> 
        <th class="col_heading level0 col1" >3/0</th> 
        <th class="col_heading level0 col2" >2/0</th> 
        <th class="col_heading level0 col3" >1/0</th> 
        <th class="col_heading level0 col4" >1</th> 
        <th class="col_heading level0 col5" >2</th> 
        <th class="col_heading level0 col6" >4</th> 
        <th class="col_heading level0 col7" >6</th> 
        <th class="col_heading level0 col8" >8</th> 
        <th class="col_heading level0 col9" >10</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_acdc8edc_1a60_11e9_b616_0a580ae941delevel0_row0" class="row_heading level0 row0" >$Gold\ R(\Omega)$</th> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow0_col0" class="data row0 col0" >0.0010939</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow0_col1" class="data row0 col1" >0.00136133</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow0_col2" class="data row0 col2" >0.0018198</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow0_col3" class="data row0 col3" >0.00229493</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow0_col4" class="data row0 col4" >0.00251666</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow0_col5" class="data row0 col5" >0.00356196</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow0_col6" class="data row0 col6" >0.00495569</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow0_col7" class="data row0 col7" >0.00769959</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow0_col8" class="data row0 col8" >0.0122728</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow0_col9" class="data row0 col9" >0.0204028</td> 
    </tr>    <tr> 
        <th id="T_acdc8edc_1a60_11e9_b616_0a580ae941delevel0_row1" class="row_heading level0 row1" >$Copper\ R(\Omega)$</th> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow1_col0" class="data row1 col0" >0.000752513</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow1_col1" class="data row1 col1" >0.000936488</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow1_col2" class="data row1 col2" >0.00125187</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow1_col3" class="data row1 col3" >0.00157873</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow1_col4" class="data row1 col4" >0.00173126</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow1_col5" class="data row1 col5" >0.00245034</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow1_col6" class="data row1 col6" >0.00340911</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow1_col7" class="data row1 col7" >0.0052967</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow1_col8" class="data row1 col8" >0.00844267</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow1_col9" class="data row1 col9" >0.0140355</td> 
    </tr>    <tr> 
        <th id="T_acdc8edc_1a60_11e9_b616_0a580ae941delevel0_row2" class="row_heading level0 row2" >$Aluminium\ R(\Omega)$</th> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow2_col0" class="data row2 col0" >0.00118965</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow2_col1" class="data row2 col1" >0.0014805</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow2_col2" class="data row2 col2" >0.00197909</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow2_col3" class="data row2 col3" >0.00249581</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow2_col4" class="data row2 col4" >0.00273695</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow2_col5" class="data row2 col5" >0.00387375</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow2_col6" class="data row2 col6" >0.00538947</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow2_col7" class="data row2 col7" >0.00837356</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow2_col8" class="data row2 col8" >0.013347</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow2_col9" class="data row2 col9" >0.0221888</td> 
    </tr>    <tr> 
        <th id="T_acdc8edc_1a60_11e9_b616_0a580ae941delevel0_row3" class="row_heading level0 row3" >$Silver\ R(\Omega)$</th> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col0" class="data row3 col0" >0.000711901</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col1" class="data row3 col1" >0.000885947</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col2" class="data row3 col2" >0.00118431</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col3" class="data row3 col3" >0.00149353</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col4" class="data row3 col4" >0.00163783</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col5" class="data row3 col5" >0.0023181</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col6" class="data row3 col6" >0.00322513</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col7" class="data row3 col7" >0.00501084</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col8" class="data row3 col8" >0.00798703</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow3_col9" class="data row3 col9" >0.013278</td> 
    </tr>    <tr> 
        <th id="T_acdc8edc_1a60_11e9_b616_0a580ae941delevel0_row4" class="row_heading level0 row4" >$Iron\ R(\Omega)$</th> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col0" class="data row4 col0" >0.00448498</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col1" class="data row4 col1" >0.00558147</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col2" class="data row4 col2" >0.00746117</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col3" class="data row4 col3" >0.00940922</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col4" class="data row4 col4" >0.0103183</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col5" class="data row4 col5" >0.014604</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col6" class="data row4 col6" >0.0203183</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col7" class="data row4 col7" >0.0315683</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col8" class="data row4 col8" >0.0503183</td> 
        <td id="T_acdc8edc_1a60_11e9_b616_0a580ae941derow4_col9" class="data row4 col9" >0.0836516</td> 
    </tr></tbody> 
</table> 
</div>





{:.input_area}
```python
res_table_ts.style.apply(highlight_min).apply(highlight_max)
```





<div markdown="0" class="output output_html">
<style  type="text/css" >
    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col0 {
            background-color:  lightblue;
            : ;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col1 {
            background-color:  lightblue;
            : ;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col2 {
            background-color:  lightblue;
            : ;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col3 {
            background-color:  lightblue;
            : ;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col4 {
            background-color:  lightblue;
            : ;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col5 {
            background-color:  lightblue;
            : ;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col6 {
            background-color:  lightblue;
            : ;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col7 {
            background-color:  lightblue;
            : ;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col8 {
            background-color:  lightblue;
            : ;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col9 {
            background-color:  lightblue;
            : ;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col0 {
            : ;
            background-color:  red;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col1 {
            : ;
            background-color:  red;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col2 {
            : ;
            background-color:  red;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col3 {
            : ;
            background-color:  red;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col4 {
            : ;
            background-color:  red;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col5 {
            : ;
            background-color:  red;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col6 {
            : ;
            background-color:  red;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col7 {
            : ;
            background-color:  red;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col8 {
            : ;
            background-color:  red;
        }    #T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col9 {
            : ;
            background-color:  red;
        }</style>  
<table id="T_ace6b4b6_1a60_11e9_b616_0a580ae941de" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >4/0</th> 
        <th class="col_heading level0 col1" >3/0</th> 
        <th class="col_heading level0 col2" >2/0</th> 
        <th class="col_heading level0 col3" >1/0</th> 
        <th class="col_heading level0 col4" >1</th> 
        <th class="col_heading level0 col5" >2</th> 
        <th class="col_heading level0 col6" >4</th> 
        <th class="col_heading level0 col7" >6</th> 
        <th class="col_heading level0 col8" >8</th> 
        <th class="col_heading level0 col9" >10</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_ace6b4b6_1a60_11e9_b616_0a580ae941delevel0_row0" class="row_heading level0 row0" >$Gold\ R(\Omega)$</th> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow0_col0" class="data row0 col0" >0.0018441</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow0_col1" class="data row0 col1" >0.00229874</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow0_col2" class="data row0 col2" >0.00307813</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow0_col3" class="data row0 col3" >0.00388586</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow0_col4" class="data row0 col4" >0.0042628</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow0_col5" class="data row0 col5" >0.0060398</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow0_col6" class="data row0 col6" >0.00840914</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow0_col7" class="data row0 col7" >0.0130738</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow0_col8" class="data row0 col8" >0.0208482</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow0_col9" class="data row0 col9" >0.0346693</td> 
    </tr>    <tr> 
        <th id="T_ace6b4b6_1a60_11e9_b616_0a580ae941delevel0_row1" class="row_heading level0 row1" >$Copper\ R(\Omega)$</th> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow1_col0" class="data row1 col0" >0.00126859</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow1_col1" class="data row1 col1" >0.00158135</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow1_col2" class="data row1 col2" >0.0021175</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow1_col3" class="data row1 col3" >0.00267316</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow1_col4" class="data row1 col4" >0.00293246</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow1_col5" class="data row1 col5" >0.0041549</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow1_col6" class="data row1 col6" >0.00578481</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow1_col7" class="data row1 col7" >0.0089937</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow1_col8" class="data row1 col8" >0.0143419</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow1_col9" class="data row1 col9" >0.0238497</td> 
    </tr>    <tr> 
        <th id="T_ace6b4b6_1a60_11e9_b616_0a580ae941delevel0_row2" class="row_heading level0 row2" >$Aluminium\ R(\Omega)$</th> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow2_col0" class="data row2 col0" >0.00200552</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow2_col1" class="data row2 col1" >0.00249996</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow2_col2" class="data row2 col2" >0.00334757</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow2_col3" class="data row2 col3" >0.004226</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow2_col4" class="data row2 col4" >0.00463593</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow2_col5" class="data row2 col5" >0.00656848</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow2_col6" class="data row2 col6" >0.00914522</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow2_col7" class="data row2 col7" >0.0142182</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow2_col8" class="data row2 col8" >0.0226731</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow2_col9" class="data row2 col9" >0.037704</td> 
    </tr>    <tr> 
        <th id="T_ace6b4b6_1a60_11e9_b616_0a580ae941delevel0_row3" class="row_heading level0 row3" >$Silver\ R(\Omega)$</th> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col0" class="data row3 col0" >0.00120013</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col1" class="data row3 col1" >0.00149601</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col2" class="data row3 col2" >0.00200323</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col3" class="data row3 col3" >0.00252889</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col4" class="data row3 col4" >0.0027742</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col5" class="data row3 col5" >0.00393066</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col6" class="data row3 col6" >0.00547261</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col7" class="data row3 col7" >0.00850833</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col8" class="data row3 col8" >0.0135679</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow3_col9" class="data row3 col9" >0.0225626</td> 
    </tr>    <tr> 
        <th id="T_ace6b4b6_1a60_11e9_b616_0a580ae941delevel0_row4" class="row_heading level0 row4" >$Iron\ R(\Omega)$</th> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col0" class="data row4 col0" >0.0075608</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col1" class="data row4 col1" >0.00942483</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col2" class="data row4 col2" >0.0126203</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col3" class="data row4 col3" >0.015932</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col4" class="data row4 col4" >0.0174775</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col5" class="data row4 col5" >0.0247632</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col6" class="data row4 col6" >0.0344775</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col7" class="data row4 col7" >0.0536025</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col8" class="data row4 col8" >0.0854775</td> 
        <td id="T_ace6b4b6_1a60_11e9_b616_0a580ae941derow4_col9" class="data row4 col9" >0.142144</td> 
    </tr></tbody> 
</table> 
</div>





{:.input_area}
```python
# Here we make a quick graphical representation of the values shown above

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("One Storey System")
for conductor in materials:
    plt.plot(wire_gauge, materials[conductor]["resistances"]["os"], "-")
    plt.plot(wire_gauge, materials[conductor]["resistances"]["os"], "o", label="%s"%(conductor))
plt.legend()
plt.xlabel(r"Cross Sectional Area of the wire $(m^{2})$")
plt.ylabel(r"$Resistance\ (\Omega)$")
    
plt.subplot(1,2,2)
plt.title("Two Storey System")
for conductor in materials:
    plt.plot(wire_gauge, materials[conductor]["resistances"]["ts"], "-")
    plt.plot(wire_gauge, materials[conductor]["resistances"]["ts"], "o", label="%s"%(conductor))
plt.legend()
plt.xlabel(r"Cross Sectional Area of the wire $(m^{2})$")
plt.ylabel(r"$Resistance\ (\Omega)$")

plt.tight_layout()
plt.show()
```



{:.output .output_png}
![png](../../../../images/essays/studentessays/stormcloud/2D_Stormcloud_Lightning-sisandst_jrevense_27_0.png)



As we can see from these tables and plots Silver has the least resistance, followed by Copper, then Gold, then Aliminium and finally Iron. We can also take note that the resistance of the wires increases as the wire gauge becomes smaller. (4/0 is the largest wire, while 4 is the smallest)



{:.input_area}
```python
# Calculates the volume of our conductors

for conductor in materials:
    vol = {}
    vol_os = (rod_gauge * one_storey_rod) + (wire_gauge * one_storey_wire)
    vol_ts = (rod_gauge * two_storey_rod) + (wire_gauge * two_storey_wire)
    
    vol["os"] = vol_os
    vol["ts"] = vol_ts
    
    materials[conductor]["vol"] = vol
```




{:.input_area}
```python
volumes_os = [materials["Gold"]["vol"]["os"]]
volumes_ts = [materials["Gold"]["vol"]["ts"]]

vol_os = pd.DataFrame(volumes_os,
                      columns=gauges, 
                      index=[r"Volume $m^{3}$"])

vol_ts = pd.DataFrame(volumes_ts,
                      columns=gauges, 
                      index=[r"Volume $m^{3}$"])
```




{:.input_area}
```python
vol_os
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>4/0</th>
      <th>3/0</th>
      <th>2/0</th>
      <th>1/0</th>
      <th>1</th>
      <th>2</th>
      <th>4</th>
      <th>6</th>
      <th>8</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Volume $m^{3}$</th>
      <td>0.000914</td>
      <td>0.000789</td>
      <td>0.000664</td>
      <td>0.000589</td>
      <td>0.000564</td>
      <td>0.000489</td>
      <td>0.000439</td>
      <td>0.000394</td>
      <td>0.000364</td>
      <td>0.000344</td>
    </tr>
  </tbody>
</table>
</div>
</div>





{:.input_area}
```python
vol_ts
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>4/0</th>
      <th>3/0</th>
      <th>2/0</th>
      <th>1/0</th>
      <th>1</th>
      <th>2</th>
      <th>4</th>
      <th>6</th>
      <th>8</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Volume $m^{3}$</th>
      <td>0.001491</td>
      <td>0.001279</td>
      <td>0.001066</td>
      <td>0.000939</td>
      <td>0.000896</td>
      <td>0.000769</td>
      <td>0.000684</td>
      <td>0.000607</td>
      <td>0.000556</td>
      <td>0.000522</td>
    </tr>
  </tbody>
</table>
</div>
</div>



Here we can see the volumes of the different setups.



{:.input_area}
```python
# Calculate the temperature rise 

currents = np.linspace(3,200,101) * (10**3) # setting up the currents with a linspace from 3-200 kA
time_strike = 65 * (10**-6) # microseconds

for conductor in materials:
    os = {} ; ts = {}
    for i in range(len(materials[conductor]["resistances"]["os"])):
        
        # Calculates the temperature rise in the one storey setup
        temp_os = temp(R = materials[conductor]["resistances"]["os"][i],
                       I = currents,
                       t = time_strike,
                       Hc = materials[conductor]["spec_heat"],
                       Vol = materials[conductor]["vol"]["os"][i],
                       rho = materials[conductor]["rho"])
        
        # Calculates the temperature rise in the Two storey setup
        temp_ts = temp(R = materials[conductor]["resistances"]["ts"][i],
                       I = currents,
                       t = time_strike,
                       Hc = materials[conductor]["spec_heat"],
                       Vol = materials[conductor]["vol"]["ts"][i],
                       rho = materials[conductor]["rho"])
        
        
        os["%s"%(i)] = temp_os
        ts["%s"%(i)] = temp_ts
    
    materials[conductor]["temps_os"] = os
    materials[conductor]["temps_ts"] = ts
```




{:.input_area}
```python
def plot(material,current):
    
    plt.figure(figsize=(16,9))
    
    print("%s: "%(material))
    
    plt.subplot(1,2,1)
    plt.title("One Storey System")
    for i in range(len(materials[material]["temps_os"])):
        plt.plot(current,materials[material]["temps_os"]["%s"%(i)], label="AWG: %s"%(gauges[i]))
    plt.legend()
    plt.xlabel(r"$Current\ (A)$")
    plt.ylabel(r"$Temperature\ Rise\ (K)$")
    
    plt.subplot(1,2,2)
    plt.title("Two Storey System")
    for i in range(len(materials[material]["temps_ts"])):
        plt.plot(current,materials[material]["temps_ts"]["%s"%(i)], label="AWG: %s"%(gauges[i]))
    plt.legend()  
    plt.xlabel(r"$Current\ (A)$")
    plt.ylabel(r"$Temperature\ Rise\ (K)$")
    
    plt.show()
        
```




{:.input_area}
```python
plot("Gold", currents)
```


{:.output .output_stream}
```
Gold: 

```


{:.output .output_png}
![png](../../../../images/essays/studentessays/stormcloud/2D_Stormcloud_Lightning-sisandst_jrevense_36_1.png)





{:.input_area}
```python
plot("Copper", currents)
```


{:.output .output_stream}
```
Copper: 

```


{:.output .output_png}
![png](../../../../images/essays/studentessays/stormcloud/2D_Stormcloud_Lightning-sisandst_jrevense_37_1.png)





{:.input_area}
```python
plot("Aluminium", currents)
```


{:.output .output_stream}
```
Aluminium: 

```


{:.output .output_png}
![png](../../../../images/essays/studentessays/stormcloud/2D_Stormcloud_Lightning-sisandst_jrevense_38_1.png)





{:.input_area}
```python
plot("Silver", currents)
```


{:.output .output_stream}
```
Silver: 

```


{:.output .output_png}
![png](../../../../images/essays/studentessays/stormcloud/2D_Stormcloud_Lightning-sisandst_jrevense_39_1.png)





{:.input_area}
```python
plot("Iron", currents)
```


{:.output .output_stream}
```
Iron: 

```


{:.output .output_png}
![png](../../../../images/essays/studentessays/stormcloud/2D_Stormcloud_Lightning-sisandst_jrevense_40_1.png)



From these plots we can see that in all cases, the temperature goes up more the thinner the wire. This is quite intuitive, since there's less material to pass the energy through.

We can also note that copper seems to be the best conductor, it's temperatures are the lowest across all wire gauges, while iron is the worst due to it's temperatures being the highest across all wire gauges.

Now we will compare the maximum temperature increases to the melting point of the PVC insulation used on the wires.



{:.input_area}
```python
# Gets the maximum temperature increase for each material and gauge

melt_point_pvc = 105 # degrees C

for conductor in materials:
    max_temps_os = {}
    max_temps_ts = {}
    
    for i in range(len(materials[conductor]["temps_os"])):
        # We add 20 degrees C to each max temp rise, since we assume we're working in an environment with
        # a temperature of 20 degrees
        max_temps_os["%s"%(i)] = np.amax(materials[conductor]["temps_os"]["%s"%(i)]) + 20 # add 20 degrees C
        max_temps_ts["%s"%(i)] = np.amax(materials[conductor]["temps_ts"]["%s"%(i)]) + 20
    
    materials[conductor]["max_temp_os"] = max_temps_os
    materials[conductor]["max_temp_ts"] = max_temps_ts
```




{:.input_area}
```python
melt_compare_os = []
melt_compare_ts = []

for conductor in materials:
    temp_list_os = []
    temp_list_ts = []
    
    for i in range(len(materials[conductor]["max_temp_os"])):
        temp_list_os.append((materials[conductor]["max_temp_os"]["%s"%(i)] / melt_point_pvc))
        temp_list_ts.append((materials[conductor]["max_temp_ts"]["%s"%(i)] / melt_point_pvc))

    melt_compare_os.append(temp_list_os)
    melt_compare_ts.append(temp_list_ts)

melt_os = pd.DataFrame(melt_compare_os,
                       columns = gauges,
                       index = mat_list)

melt_ts = pd.DataFrame(melt_compare_ts,
                       columns = gauges,
                       index = mat_list)
```




{:.input_area}
```python
melt_os.style.format("{:.2%}").apply(highlight_min).apply(highlight_max).applymap(color_val)
```





<div markdown="0" class="output output_html">
<style  type="text/css" >
    #T_af633598_1a60_11e9_b616_0a580ae941derow0_col0 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow0_col1 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow0_col2 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow0_col3 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow0_col4 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow0_col5 {
            : ;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow0_col6 {
            : ;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow0_col7 {
            : ;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow0_col8 {
            : ;
            : ;
            color:  orange;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow0_col9 {
            : ;
            : ;
            color:  darkorange;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow1_col0 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow1_col1 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow1_col2 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow1_col3 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow1_col4 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow1_col5 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow1_col6 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow1_col7 {
            background-color:  lightblue;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow1_col8 {
            background-color:  lightblue;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow1_col9 {
            background-color:  lightblue;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow2_col0 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow2_col1 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow2_col2 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow2_col3 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow2_col4 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow2_col5 {
            : ;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow2_col6 {
            : ;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow2_col7 {
            : ;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow2_col8 {
            : ;
            : ;
            color:  orange;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow2_col9 {
            : ;
            : ;
            color:  darkorange;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow3_col0 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow3_col1 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow3_col2 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow3_col3 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow3_col4 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow3_col5 {
            : ;
            : ;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow3_col6 {
            : ;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow3_col7 {
            : ;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow3_col8 {
            : ;
            : ;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow3_col9 {
            : ;
            : ;
            color:  orange;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow4_col0 {
            : ;
            background-color:  red;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow4_col1 {
            : ;
            background-color:  red;
            color:  green;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow4_col2 {
            : ;
            background-color:  red;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow4_col3 {
            : ;
            background-color:  red;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow4_col4 {
            : ;
            background-color:  red;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow4_col5 {
            : ;
            background-color:  red;
            color:  black;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow4_col6 {
            : ;
            background-color:  red;
            color:  orange;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow4_col7 {
            : ;
            background-color:  red;
            color:  orange;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow4_col8 {
            : ;
            background-color:  red;
            color:  pink;
        }    #T_af633598_1a60_11e9_b616_0a580ae941derow4_col9 {
            : ;
            background-color:  red;
            color:  pink;
        }</style>  
<table id="T_af633598_1a60_11e9_b616_0a580ae941de" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >4/0</th> 
        <th class="col_heading level0 col1" >3/0</th> 
        <th class="col_heading level0 col2" >2/0</th> 
        <th class="col_heading level0 col3" >1/0</th> 
        <th class="col_heading level0 col4" >1</th> 
        <th class="col_heading level0 col5" >2</th> 
        <th class="col_heading level0 col6" >4</th> 
        <th class="col_heading level0 col7" >6</th> 
        <th class="col_heading level0 col8" >8</th> 
        <th class="col_heading level0 col9" >10</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_af633598_1a60_11e9_b616_0a580ae941delevel0_row0" class="row_heading level0 row0" >$Gold\ R(\Omega)$</th> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow0_col0" class="data row0 col0" >20.27%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow0_col1" class="data row0 col1" >20.81%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow0_col2" class="data row0 col2" >21.85%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow0_col3" class="data row0 col3" >23.03%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow0_col4" class="data row0 col4" >23.60%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow0_col5" class="data row0 col5" >26.49%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow0_col6" class="data row0 col6" >30.57%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow0_col7" class="data row0 col7" >39.00%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow0_col8" class="data row0 col8" >53.47%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow0_col9" class="data row0 col9" >79.60%</td> 
    </tr>    <tr> 
        <th id="T_af633598_1a60_11e9_b616_0a580ae941delevel0_row1" class="row_heading level0 row1" >$Copper\ R(\Omega)$</th> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow1_col0" class="data row1 col0" >19.65%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow1_col1" class="data row1 col1" >19.92%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow1_col2" class="data row1 col2" >20.43%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow1_col3" class="data row1 col3" >21.01%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow1_col4" class="data row1 col4" >21.30%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow1_col5" class="data row1 col5" >22.72%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow1_col6" class="data row1 col6" >24.74%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow1_col7" class="data row1 col7" >28.90%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow1_col8" class="data row1 col8" >36.05%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow1_col9" class="data row1 col9" >48.96%</td> 
    </tr>    <tr> 
        <th id="T_af633598_1a60_11e9_b616_0a580ae941delevel0_row2" class="row_heading level0 row2" >$Aluminium\ R(\Omega)$</th> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow2_col0" class="data row2 col0" >20.39%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow2_col1" class="data row2 col1" >20.99%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow2_col2" class="data row2 col2" >22.13%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow2_col3" class="data row2 col3" >23.43%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow2_col4" class="data row2 col4" >24.06%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow2_col5" class="data row2 col5" >27.24%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow2_col6" class="data row2 col6" >31.74%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow2_col7" class="data row2 col7" >41.01%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow2_col8" class="data row2 col8" >56.94%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow2_col9" class="data row2 col9" >85.71%</td> 
    </tr>    <tr> 
        <th id="T_af633598_1a60_11e9_b616_0a580ae941delevel0_row3" class="row_heading level0 row3" >$Silver\ R(\Omega)$</th> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow3_col0" class="data row3 col0" >19.82%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow3_col1" class="data row3 col1" >20.16%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow3_col2" class="data row3 col2" >20.81%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow3_col3" class="data row3 col3" >21.55%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow3_col4" class="data row3 col4" >21.92%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow3_col5" class="data row3 col5" >23.73%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow3_col6" class="data row3 col6" >26.30%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow3_col7" class="data row3 col7" >31.61%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow3_col8" class="data row3 col8" >40.72%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow3_col9" class="data row3 col9" >57.17%</td> 
    </tr>    <tr> 
        <th id="T_af633598_1a60_11e9_b616_0a580ae941delevel0_row4" class="row_heading level0 row4" >$Iron\ R(\Omega)$</th> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow4_col0" class="data row4 col0" >22.40%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow4_col1" class="data row4 col1" >23.88%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow4_col2" class="data row4 col2" >26.72%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow4_col3" class="data row4 col3" >29.96%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow4_col4" class="data row4 col4" >31.54%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow4_col5" class="data row4 col5" >39.44%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow4_col6" class="data row4 col6" >50.66%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow4_col7" class="data row4 col7" >73.76%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow4_col8" class="data row4 col8" >113.45%</td> 
        <td id="T_af633598_1a60_11e9_b616_0a580ae941derow4_col9" class="data row4 col9" >185.10%</td> 
    </tr></tbody> 
</table> 
</div>





{:.input_area}
```python
melt_ts.style.format("{:.2%}").apply(highlight_min).apply(highlight_max).applymap(color_val)
```





<div markdown="0" class="output output_html">
<style  type="text/css" >
    #T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col0 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col1 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col2 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col3 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col4 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col5 {
            : ;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col6 {
            : ;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col7 {
            : ;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col8 {
            : ;
            : ;
            color:  orange;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col9 {
            : ;
            : ;
            color:  darkorange;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col0 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col1 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col2 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col3 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col4 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col5 {
            background-color:  lightblue;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col6 {
            background-color:  lightblue;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col7 {
            background-color:  lightblue;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col8 {
            background-color:  lightblue;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col9 {
            background-color:  lightblue;
            : ;
            color:  orange;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col0 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col1 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col2 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col3 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col4 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col5 {
            : ;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col6 {
            : ;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col7 {
            : ;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col8 {
            : ;
            : ;
            color:  orange;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col9 {
            : ;
            : ;
            color:  darkorange;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col0 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col1 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col2 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col3 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col4 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col5 {
            : ;
            : ;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col6 {
            : ;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col7 {
            : ;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col8 {
            : ;
            : ;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col9 {
            : ;
            : ;
            color:  orange;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col0 {
            : ;
            background-color:  red;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col1 {
            : ;
            background-color:  red;
            color:  green;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col2 {
            : ;
            background-color:  red;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col3 {
            : ;
            background-color:  red;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col4 {
            : ;
            background-color:  red;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col5 {
            : ;
            background-color:  red;
            color:  black;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col6 {
            : ;
            background-color:  red;
            color:  orange;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col7 {
            : ;
            background-color:  red;
            color:  darkorange;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col8 {
            : ;
            background-color:  red;
            color:  pink;
        }    #T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col9 {
            : ;
            background-color:  red;
            color:  pink;
        }</style>  
<table id="T_af6e95be_1a60_11e9_b616_0a580ae941de" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >4/0</th> 
        <th class="col_heading level0 col1" >3/0</th> 
        <th class="col_heading level0 col2" >2/0</th> 
        <th class="col_heading level0 col3" >1/0</th> 
        <th class="col_heading level0 col4" >1</th> 
        <th class="col_heading level0 col5" >2</th> 
        <th class="col_heading level0 col6" >4</th> 
        <th class="col_heading level0 col7" >6</th> 
        <th class="col_heading level0 col8" >8</th> 
        <th class="col_heading level0 col9" >10</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_af6e95be_1a60_11e9_b616_0a580ae941delevel0_row0" class="row_heading level0 row0" >$Gold\ R(\Omega)$</th> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col0" class="data row0 col0" >20.31%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col1" class="data row0 col1" >20.88%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col2" class="data row0 col2" >22.00%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col3" class="data row0 col3" >23.28%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col4" class="data row0 col4" >23.91%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col5" class="data row0 col5" >27.07%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col6" class="data row0 col6" >31.61%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col7" class="data row0 col7" >41.04%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col8" class="data row0 col8" >57.33%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow0_col9" class="data row0 col9" >86.86%</td> 
    </tr>    <tr> 
        <th id="T_af6e95be_1a60_11e9_b616_0a580ae941delevel0_row1" class="row_heading level0 row1" >$Copper\ R(\Omega)$</th> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col0" class="data row1 col0" >19.67%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col1" class="data row1 col1" >19.95%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col2" class="data row1 col2" >20.50%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col3" class="data row1 col3" >21.14%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col4" class="data row1 col4" >21.45%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col5" class="data row1 col5" >23.01%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col6" class="data row1 col6" >25.25%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col7" class="data row1 col7" >29.91%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col8" class="data row1 col8" >37.96%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow1_col9" class="data row1 col9" >52.54%</td> 
    </tr>    <tr> 
        <th id="T_af6e95be_1a60_11e9_b616_0a580ae941delevel0_row2" class="row_heading level0 row2" >$Aluminium\ R(\Omega)$</th> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col0" class="data row2 col0" >20.44%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col1" class="data row2 col1" >21.07%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col2" class="data row2 col2" >22.29%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col3" class="data row2 col3" >23.70%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col4" class="data row2 col4" >24.40%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col5" class="data row2 col5" >27.88%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col6" class="data row2 col6" >32.88%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col7" class="data row2 col7" >43.26%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col8" class="data row2 col8" >61.19%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow2_col9" class="data row2 col9" >93.70%</td> 
    </tr>    <tr> 
        <th id="T_af6e95be_1a60_11e9_b616_0a580ae941delevel0_row3" class="row_heading level0 row3" >$Silver\ R(\Omega)$</th> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col0" class="data row3 col0" >19.84%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col1" class="data row3 col1" >20.20%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col2" class="data row3 col2" >20.90%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col3" class="data row3 col3" >21.71%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col4" class="data row3 col4" >22.11%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col5" class="data row3 col5" >24.10%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col6" class="data row3 col6" >26.96%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col7" class="data row3 col7" >32.89%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col8" class="data row3 col8" >43.15%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow3_col9" class="data row3 col9" >61.74%</td> 
    </tr>    <tr> 
        <th id="T_af6e95be_1a60_11e9_b616_0a580ae941delevel0_row4" class="row_heading level0 row4" >$Iron\ R(\Omega)$</th> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col0" class="data row4 col0" >22.51%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col1" class="data row4 col1" >24.08%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col2" class="data row4 col2" >27.13%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col3" class="data row4 col3" >30.64%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col4" class="data row4 col4" >32.37%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col5" class="data row4 col5" >41.05%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col6" class="data row4 col6" >53.50%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col7" class="data row4 col7" >79.35%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col8" class="data row4 col8" >124.03%</td> 
        <td id="T_af6e95be_1a60_11e9_b616_0a580ae941derow4_col9" class="data row4 col9" >205.00%</td> 
    </tr></tbody> 
</table> 
</div>



These tables show how close to the melting point of the PVC insulation that the cables got as a percentage of the melting point of the PVC insulation.

# Conclusion

## Lightning Rod Simulation

From our simulations we see that having a lightning rod present will in the majority of the cases cause the lightning to strike the rod. As we noted earlier the rod will make a 45 degree "cone of protection" under it, this means that one should take this into account when setting up lightning rods for practical applications.

## Lightning Rod Materials

We chose to test a few metals for these calculations, Gold, Copper, Aluminium, Silver and Iron were our materials of choice.

From our calculations of resistance silver seemed to be the best candidate, due to it having the lowest resistance across all wire gauges. The difference in resistance between silver and the second best material(copper) was small, thus both are good conductors.

However when it came to the final result with the temperature increase, copper was clearly the best candidate, due to it having the lowest temperature increase across all wire gauges. The difference in temperature increase was not that large when it came to the bigger wire gauges, but as the wires became smaller the differences became significantly greater.

Most of the materials were well within safe temperature increase across the entire range of current, iron however was not for the thinnest gauges, aluminium and gold got dangerously close for the thinnest gauge.

For all applications copper seems to be the best alternative.

## Words of wisdom:

Check your units, always, always, always, check your units!
