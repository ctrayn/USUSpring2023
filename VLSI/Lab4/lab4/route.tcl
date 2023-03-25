 
#set the relevant paths
source env.tcl

#load placed design, library, design constraints
restoreDesign s35932.placed.enc.dat s35932_bench

#Timing Analysis 
##Ran after placement in place.tcl
#timedesign -preCTS -outDir prectsTimingReports

#Optimization
optdesign -preCTS -outDir prectsOptTimingReports

#Clock Tree Synthesis
create_ccopt_clock_tree_spec
ccopt_design

#Timing Analysis and Optimization
timeDesign -postCTS -outDir postctsTimingReports
optDesign -postCTS -outDir postctsOptTimngReports

#perform global routing
globalRoute
#win 
#fit

#perform detail routing
setNanoRouteMode -routeWithTimingDriven true
detailRoute

#globalDetailRoute
setAnalysisMode -analysisType onChipVariation
timeDesign -postRoute -outDir postrouteTimingReports
optDesign -postRoute -outDir postrouteOptTimingReports

#Verification
verify_drc 
verifyConnectivity

#Commands to reduce the violations during routing
##Larger designs might require running the below commands multipler times
#editDelete -regular_wire_with_drc
#globalDetailRoute
#verify_drc

# Report Output Power
report_power -o outputPower

#exit

