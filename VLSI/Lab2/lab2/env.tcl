# set the following parameter to point to the 
# location where the design data is located.
# Design data includes gate level netlist (.v file), design floorplan,
# design configuration file (.conf file), timing constraint file (.sdc)
set designDir ./s35932

# set the following variable to point to the location where the
# libraries are located. 
# Timing Library - .lib
# Synopsys database format - .db
# LEF file
# Capacitance Table (.cap)
set libDir /opt/cadence_roy/library

# set the following variable to point to the directory 
# where you want to store the intermediate results.
# The intermediate results include the design saved at
# various stages of the flow. e.g You may want to save the 
# design after placement or routing or clock tree synthesis
# or in between while doing timing optimizations.
set outDir ./outputDir
