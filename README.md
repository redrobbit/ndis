# Natural Disaster Information System (NDIS) – Geohazard Mission Planning

This repository documents the end-to-end geospatial preprocessing and decision logic pipeline used in the Natural Disaster Information System (NDIS). The system supports drone-based (RPAS) hazard survey planning by integrating multi-source geohazard datasets, performing spatial analysis, and applying both rule-based and machine learning-based decision workflows.

>  Developed as part of a PhD thesis at Queen's University, Canada  
>  Tools: ArcGIS Pro, Python (arcpy), FastAPI, Azure App Services, ArcGIS Experience Builder

Dataset Used in this research are derived from:
1. VOLCANO:
 
Volcano data retrieved from GVP, GVP VOTW, Smithsonian Institution and Significant Volcano Eruption from NGDC/WDS.
 
Global Volcanism Program:
Citation: Global Volcanism Program, 2013. Volcanoes of the World, v. 4.11.0 (08 Jul 2022). Venzke, E (ed.). Smithsonian Institution. Downloaded 13 Jul 2022. https://doi.org/10.5479/si.GVP.VOTW4-2013. 
 
GVP VOTW:
Further info: https://volcano.si.edu/database/webservices.cfm 
Service Layer: https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=1.0.0&request=describefeaturetype&typeName=GVP-VOTW:E3WebApp_HoloceneVolcanoes 
 
Significant Volcano Eruption: 
Citation: National Geophysical Data Center / World Data Service (NGDC/WDS): Significant Earthquake Database. National Geophysical Data Center, NOAA. doi:10.7289/V5TD9V7K
 

2. LANDSLIDE:
Landslide data retrieved from:
 
Title: Global Landslide Catalog | Type: Feature Service | Owner: krolikie@unhcr.org_unhcr https://maps.nccs.nasa.gov/arcgis/apps/MapAndAppGallery/index.html?appid=574f26408683485799d02e857e5d9521
 
 
Citation: Kirschbaum, D.B., Stanley, T., & Zhou, Y. (2015). Spatial and temporal analysis of a global landslide catalog. Geomorphology, 249, 4-15. doi:10.1016/j.geomorph.2015.03.016 Kirschbaum, D.B., Adler, R., Hong, Y., Hill, S., & Lerner-Lam, A. (2010). A global landslide catalog for hazard applications: method, results, and limitations. Natural Hazards, 52, 561-575. doi:10.1007/s11069-009-9401-4 Further info: https://gpm.nasa.gov/landslides/data.html
 

3. TSUNAMI
Data retrieved from NCEI NOAA - Global Historical Tsunami Database
 
Citation: National Geophysical Data Center / World Data Service: NCEI/WDS Global Historical Tsunami Database. NOAA National Centers for Environmental Information. doi:10.7289/V5PN93H7 [4 August 2023] 
Further info: https://ngdc.noaa.gov/hazard/hazards.shtml
Documentation: https://data.noaa.gov/metaview/page?xml=NOAA/NESDIS/NGDC/MGG/Hazards/iso/xml/G02151.xml&view=getDataView 
 
Layer info: https://www.arcgis.com/home/item.html?id=5a44c3d4d465498993120b70ab568876
 

4. FAULT
Fault original dataset is retrieved from GEM Global Active Faults Database (GAF-DB)
 
Citation: 
The GEM GAF-DB has been published in Earthquake Spectra. Styron, Richard, and Marco Pagani. “The GEM Global Active Faults Database.” Earthquake Spectra, vol. 36, no. 1_suppl, Oct. 2020, pp. 160–180, doi:10.1177/8755293020944182.
The link to the publication is here: https://journals.sagepub.com/doi/abs/10.1177/8755293020944182
Documentation: https://github.com/GEMScienceTools/gem-global-active-faults
 


5. EARTHQUAKE:
Earthquake dats retrieved from the following datasets.
 
Historical Parts:
GHEC Catalog
GEM provides Global Historical Earthquake Catalogue (GHEC) from 1000 to 1903 (Albini, 2014). (https://platform.openquake.org/maps/80/download)
 
SHEEC (SHARE European Earthquake Catalogue)
SHEEC catalogue covers the year 1000-1899 for Europe region specifically (Stucchi et al., 2012). 
The data could be downloaded at https://www.emidius.eu/SHEEC/sheec_1000_1899.html.
 
 
Instrumental Part:
ISC Bulletin/ISC Global
Data period 1900-2023. International Seismological Centre (2023), On-line Bulletin, https://doi.org/10.31905/D808B830
(https://www.isc.ac.uk/iscbulletin/search/catalogue/)
Compiled from 573 around the world. For more information about the agency (https://www.isc.ac.uk/iscbulletin/agencies/).
 
ISC-GEM Catalogue
The ISC-GEM Global Instrumental Earthquake Catalogue (for data period 1904-2016).
(https://www.isc.ac.uk/iscgem/)
 


6. NUCLEAR POWER PLANT
Nuclear Power Plant data retrieved from Global Energy Monitor.
Copyright © Global Energy Monitor. Global Nuclear Power Tracker, July 2024 release. Distributed under a Creative Commons Attribution 4.0 International License.
"Global Energy Monitor, Global Nuclear Power Tracker, July 2024 release" (See the CC license for attribution requirements if sharing or adapting the data set.)
https://globalenergymonitor.org/projects/global-nuclear-power-tracker/
 
 
Processing Note: 
Processed by Robiah Al Wardah
Clipped by Region using exclusive economic zone (EEZ):
Retrieved from Maritime Boundaries and Exclusive Economic Zones (200NM), version 12 https://www.marineregions.org/. https://doi.org/10.14284/632
Citation: Flanders Marine Institute (2024). Union of the ESRI Country shapefile and the Exclusive Economic Zones (version 4). Available online at https://www.marineregions.org/. https://doi.org/10.14284/698. Consulted on 2025-02-20 Further info: https://www.marineregions.org/downloads.php#unioneezcountry
 
Distance or Near Analysis using road layer: 
Global Roads Inventory Project (GRIP) database Citation: Meijer, J.R., Huijbregts, M.A.J., Schotten, C.G.J. and Schipper, A.M. (2018): Global patterns of current and future road infrastructure. Environmental Research Letters, 13-064006. Data is available at www.globio.info. Accessed 14 July 2022. 
 
Population by radius 30 km: 
Center For International Earth Science Information Network-CIESIN-Columbia University. (2018). Gridded Population of the World, Version 4 (GPWv4): Population Count, Revision 11 (Version 4.11) [Data set]. Palisades, NY: NASA Socioeconomic Data and Applications Center (SEDAC). https://doi.org/10.7927/H4JW8BX5. Accessed 24-03-2025. https://data.naturalcapitalproject.stanford.edu/dataset/sts-f899bfb8c4051511f0cf31d237beb485013b21fdee06c0c6e439168cfb8088f0
