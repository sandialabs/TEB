# TEB
Tiered Energy in Buildings - This is the initial release of a tool that is under construction. It has some limited testing but is mostly useful for some very specific cases.

The tool runs a "Tiered Analysis" for R5C1 models using the RC Building Simulator python package (not installable except by the source)

https://github.com/architecture-building-systems/RC_BuildingSimulator

Tiered Analysis is defined as a set of analyses on buildings where circuits in the building are on different Tiers. Tier 1 is the highest tier and represents 
loads that must be sustained during power outages. Tier 2 are less critical and Tier 3 is the least critical set of loads. The final Tier is non-critical. 

The spreadsheets in the LoadsData folder is the input file for the Tiered Analysis. More documentation of this tool will be initiated as it is used more. 
