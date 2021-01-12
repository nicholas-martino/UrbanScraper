import pandas as pd
import geopandas as gpd
from elementslab.Analyst import GeoBoundary
from elementslab._0_Variables import radius, network_layers

city = GeoBoundary('Metro Vancouver, British Columbia')

# Join housing prices from craigslist_housing to land_parcels
cl_housing = gpd.read_file(city.gpkg, layer='craigslist_rent')
# parcels = gpd.read_file('https://opendata.vancouver.ca/explore/dataset/property-parcel-polygons/download/?format=geojson&timezone=America/Los_Angeles&lang=en')
parcels = gpd.read_file(city.gpkg, layer='land_assessment_parcels')
[parcels.drop(i, axis=1, inplace=True) for i in ['index_left', 'index_right'] if i in parcels.columns]
parcels_geom = parcels['geometry']
parcels = gpd.sjoin(parcels, cl_housing, how='left').groupby('site_id').mean()
parcels['geometry'] = parcels_geom
parcels = gpd.GeoDataFrame(parcels, geometry='geometry')
parcels.to_file(city.gpkg, layer='land_parcels_priced')
parcels_priced = parcels.loc[parcels['price_sqft'][[not b for b in parcels['price_sqft'].isna()]].index, :]

# Run centrality indicators
city.centrality(run=False, axial=True, layer='network_walk')
city.centrality(run=False, osm=True, layer='network_drive')

# Run topography indicator
city.node_elevation(run=False)

# Run network analysis
filter_min = {'population density per square kilometre, 2016': 300}
network_analysis = city.network_analysis(
    prefix='aff_arbutus',
    run=True,
    service_areas=radius,
    sample_layer='land_parcels_vancouver',
    decays=['flat'],
    filter_min=filter_min,
    aggregated_layers=network_layers,
    keep=['dauid', 'price_sqft', 'geometry'])
network_analysis = network_analysis.to_crs(3857)
network_analysis.to_file('aff_arbutus.geojson', driver='GeoJSON')
