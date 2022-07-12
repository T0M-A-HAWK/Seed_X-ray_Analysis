# Seed_X-ray_Analysis_README

- This algorithm uses PtQt 5 to analyse X-ray images of native seeds to determine their properties and viability. 
- The algorithm is ran with the Python file of whichever version is to be used. There must be a ui file in the same folder as the Python file.
- Presets can be used to automatically set parameters for an image, preset options will be generated if there is a file in the same folder with a matching name in the code which must contain values for each parameter for the desired species.

Features of Version 1.4:
- Generates the parameter preset menu from code based on the csv file instead of being pre-populated in QT Designer in 1.3.
- Can swap viability of a seed (ie. from viable to non-viable or vice versa) by left-clicking on it after the viable seeds have been counted.
