import os
import timeit
import zipfile

import geopandas as gpd
import pandas as pd
from UrbanMobility.Analyst import Network
from shapely.geometry import Point


class BritishColumbia:
    def __init__(self, cities, directory='/Volumes/Samsung_T5/Databases/'):
        self.cities = [Network(f"{city}, British Columbia", directory=directory) for city in cities]
        self.dir = directory
        print("\n### Class British Columbia created ###\n\n")

    def update_databases(self, icbc=True):

        # Check if ICBC crash data exists and join it from ICBC database if not
        if icbc:
            for city in self.cities:

                try:
                    city.crashes = gpd.read_file(f"{self.dir}{city}.gpkg", layer='network_accidents')
                    print(city + ' ICBC data read from database')
                except:
                    source = 'https://public.tableau.com/profile/icbc#!/vizhome/LowerMainlandCrashes/LMDashboard'
                    print('Adding ICBC crash data to ' + city + ' database')
                    df = city.merge_csv(f"{self.dir}ICBC/")
                    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
                    gdf = gpd.GeoDataFrame(df, geometry=geometry)
                    gdf.crs = 4326
                    gdf.to_crs(epsg=city.crs, inplace=True)
                    city.boundary.to_crs(epsg=city.crs, inplace=True)
                    matches = gpd.sjoin(gdf, city.boundary, op='within')
                    try: matches.to_file(city.gpkg, layer='icbc_accidents', driver='GPKG')
                    except: pass

    # BC Assessment
    def aggregate_bca_from_field(self, run=True, join=True, classify=True, inventory_dir='', geodatabase_dir=''):
        if run:
            for city in self.cities:
                start_time = timeit.default_timer()
                if join:
                    print('> Geoprocessing BC Assessment data from JUROL number')
                    inventory = inventory_dir
                    df = pd.read_csv(inventory)

                    # Load and process Roll Number field on both datasets
                    print("> Reading GeoDataFrames")
                    gdf = gpd.read_file(geodatabase_dir, layer='ASSESSMENT_FABRIC', dtype='unicode', error_bad_lines=False)
                    boundary = gpd.read_file(city.gpkg, layer='land_municipal_boundary', driver='GPKG')

                    # Reproject coordinate system
                    print("> Reprojecting to UTM")
                    gdf.crs = 3005
                    gdf.to_crs(epsg=city.crs, inplace=True)
                    boundary.to_crs(epsg=city.crs, inplace=True)

                    # Create spatial index and perform join
                    print("> Joining polygons within city boundary")
                    s_index = gdf.sindex
                    gdf = gpd.sjoin(gdf, boundary, op='within')

                    # Change feature types
                    gdf['JUROL'] = gdf['JUROL'].astype(str)
                    gdf = gdf[gdf.geometry.area > 71]
                    df['JUR'] = df['JUR'].astype(int).astype(str)
                    df['ROLL_NUM'] = df['ROLL_NUM'].astype(str)
                    df['JUROL'] = df['JUR'] + df['ROLL_NUM']
                    print(f'BCA spatial layer loaded with {len(gdf)} parcels')

                    # Merge by JUROL field
                    merged = pd.merge(gdf, df, on='JUROL')
                    full_gdfs = {'0z': merged}
                    print(f": {len(full_gdfs['0z'])}")

                    # Test merge with variations of JUROL
                    for i in range(1, 7):
                        strings = []
                        for n in range(i):
                            strings.append('0')
                        string = str(''.join(strings))
                        df[string + 'z'] = string
                        df['JUROL'] = df['JUR'] + string + df['ROLL_NUM']
                        full_gdf = pd.merge(gdf, df, on='JUROL')
                        full_gdf.drop([string + 'z'], axis=1)
                        if len(full_gdf) > 0:
                            full_gdfs[str(i) + 'z'] = full_gdf
                        print(f"string: {len(full_gdf)}")

                    # Merge and export spatial and non-spatial datasets
                    out_gdf = pd.concat(full_gdfs.values(), ignore_index=True)
                    out_gdf.to_file(city.gpkg, driver='GPKG', layer='land_assessment_fabric')
                    print(len(out_gdf))

                if classify:
                    print("> Classifying land uses and parcel categories from BC Assessment")
                    if not join: out_gdf = gpd.read_file(city.gpkg, driver='GPKG', layer='land_assessment_fabric')

                    # Reclassify land uses for BC Assessment data
                    uses = {
                        'residential': ['000 - Single Family Dwelling',
                                            '030 - Strata-Lot Residence (Condominium)',
                                            '032 - Residential Dwelling with Suite',
                                            '033 - Duplex, Non-Strata Side by Side or Front / Back',
                                            '034 - Duplex, Non-Strata Up / Down',
                                            '035 - Duplex, Strata Side by Side',
                                            '036 - Duplex, Strata Front / Back',
                                            '037 - Manufactured Home (Within Manufactured Home Park)',
                                            '038 - Manufactured Home (Not In Manufactured Home Park)',
                                            '039 - Row Housing (Single Unit Ownership)',
                                            '040 - Seasonal Dwelling',
                                            '041 - Duplex, Strata Up / Down',
                                            '047 - Triplex',
                                            '049 - Fourplex',
                                            '050 - Multi-Family (Apartment Block)',
                                            '052 - Multi-Family (Garden Apartment & Row Housing)',
                                            '053 - Multi-Family (Conversion)',
                                            '054 - Multi-Family (High-Rise)', '055 - Multi-Family (Minimal Commercial)',
                                            '056 - Multi-Family (Residential Hotel)',
                                            '057 - Stratified Rental Townhouse',
                                            '058 - Stratified Rental Apartment (Frame Construction)',
                                            '059 - Stratified Rental Apartment (Hi-Rise Construction)',
                                            '060 - 2 Acres Or More (Single Family Dwelling, Duplex)',
                                            '062 - 2 Acres Or More (Seasonal Dwelling)',
                                            '063 - 2 Acres Or More (Manufactured Home)',
                                            '234 - Manufactured Home Park',
                                            '284 - Seniors Strata - Care',
                                            '285 - Seniors Licensed Care',
                                            '286 - Seniors Independent & Assisted Living'],
                        'hospitality': ['042 - Strata-Lot Seasonal Dwelling (Condominium)',
                                            '230 - Hotel',
                                            '232 - Motel & Auto Court',
                                            '233 - Individual Strata Lot (Hotel/Motel)',
                                            '237 - Bed & Breakfast Operation 4 Or More Units',
                                            '239 - Bed & Breakfast Operation Less Than 4 Units',
                                            '238 - Seasonal Resort'],
                        'retail': ['202 - Store(S) And Living Quarters',
                                       '206 - Neighbourhood Store',
                                       '209 - Shopping Centre (Neighbourhood)',
                                       '211 - Shopping Centre (Community)',
                                       '212 - Department Store - Stand Alone',
                                       '213 - Shopping Centre (Regional)',
                                       '214 - Retail Strip',
                                       '215 - Food Market',
                                       '216 - Commercial Strata-Lot',
                                       '225 - Convenience Store/Service Station'],
                        'entertainment': ['236 - Campground (Commercial)',
                                              '250 - Theatre Buildings',
                                              '254 - Neighbourhood Pub',
                                              '256 - Restaurant Only',
                                              '257 - Fast Food Restaurants',
                                              '266 - Bowling Alley',
                                              '270 - Hall (Community, Lodge, Club, Etc.)',
                                              '280 - Marine Facilities (Marina)',
                                              '600 - Recreational & Cultural Buildings (Includes Curling',
                                              '610 - Parks & Playing Fields',
                                              '612 - Golf Courses (Includes Public & Private)',
                                              '614 - Campgrounds',
                                              '654 - Recreational Clubs, Ski Hills',
                                              '660 - Land Classified Recreational Used For'],
                        'civic': ['210 - Bank',
                                      '620 - Government Buildings (Includes Courthouse, Post Office',
                                      '625 - Garbage Dumps, Sanitary Fills, Sewer Lagoons, Etc.',
                                      '630 - Works Yards',
                                      '634 - Government Research Centres (Includes Nurseries &',
                                      '640 - Hospitals (Nursing Homes Refer To Commercial Section).',
                                      '642 - Cemeteries (Includes Public Or Private).',
                                      '650 - Schools & Universities, College Or Technical Schools',
                                      '652 - Churches & Bible Schools'],
                        'office': ['203 - Stores And/Or Offices With Apartments',
                                       '204 - Store(S) And Offices',
                                       '208 - Office Building (Primary Use)']
                    }
                    elab_uses = {
                        'SFD': [
                            '000 - Single Family Dwelling',
                            '030 - Strata-Lot Residence (Condominium)',
                            '032 - Residential Dwelling with Suite',
                            '037 - Manufactured Home (Within Manufactured Home Park)',
                            '038 - Manufactured Home (Not In Manufactured Home Park)',
                            '040 - Seasonal Dwelling',
                            '042 - Strata-Lot Seasonal Dwelling (Condominium)',
                            '060 - 2 Acres Or More (Single Family Dwelling, Duplex)',
                            '062 - 2 Acres Or More (Seasonal Dwelling)',
                            '063 - 2 Acres Or More (Manufactured Home)',
                            '234 - Manufactured Home Park',
                        ],
                        'SFA': [
                            '033 - Duplex, Non-Strata Side by Side or Front / Back',
                            '034 - Duplex, Non-Strata Up / Down',
                            '035 - Duplex, Strata Side by Side',
                            '036 - Duplex, Strata Front / Back',
                            '039 - Row Housing (Single Unit Ownership)',
                            '041 - Duplex, Strata Up / Down',
                            '047 - Triplex',
                            '049 - Fourplex',
                            '284 - Seniors Strata - Care',
                            '285 - Seniors Licensed Care',
                            '286 - Seniors Independent & Assisted Living',
                        ],
                        'MFL': [
                            '050 - Multi-Family (Apartment Block)',
                            '052 - Multi-Family (Garden Apartment & Row Housing)',
                            '053 - Multi-Family (Conversion)',
                            '057 - Stratified Rental Townhouse',
                        ],
                        'MFH': [
                            '054 - Multi-Family (High-Rise)',
                            '056 - Multi-Family (Residential Hotel)',
                            '058 - Stratified Rental Apartment (Frame Construction)',
                            '059 - Stratified Rental Apartment (Hi-Rise Construction)',
                        ],
                        'MX': [
                            '055 - Multi-Family (Minimal Commercial)',
                            '203 - Stores And/Or Offices With Apartments',
                        ],
                        'CM': [
                            '202 - Store(S) And Living Quarters',
                            '204 - Store(S) And Offices',
                            '206 - Neighbourhood Store',
                            '208 - Office Building (Primary Use)',
                            '209 - Shopping Centre (Neighbourhood)',
                            '210 - Bank',
                            '211 - Shopping Centre (Community)',
                            '212 - Department Store - Stand Alone',
                            '213 - Shopping Centre (Regional)',
                            '214 - Retail Strip',
                            '215 - Food Market',
                            '216 - Commercial Strata-Lot',
                            '225 - Convenience Store/Service Station',
                            '230 - Hotel',
                            '232 - Motel & Auto Court',
                            '233 - Individual Strata Lot (Hotel/Motel)',
                            '237 - Bed & Breakfast Operation 4 Or More Units',
                            '239 - Bed & Breakfast Operation Less Than 4 Units',
                            '238 - Seasonal Resort'
                            '250 - Theatre Buildings',
                            '254 - Neighbourhood Pub',
                            '256 - Restaurant Only',
                            '257 - Fast Food Restaurants',
                            '266 - Bowling Alley',
                            '270 - Hall (Community, Lodge, Club, Etc.)',
                            '280 - Marine Facilities (Marina)',
                            '600 - Recreational & Cultural Buildings (Includes Curling',
                        ],
                        'OS': [
                            '236 - Campground (Commercial)',
                            '610 - Parks & Playing Fields',
                            '612 - Golf Courses (Includes Public & Private)',
                            '614 - Campgrounds',
                            '654 - Recreational Clubs, Ski Hills',
                            '660 - Land Classified Recreational Used For'
                        ],
                        'CV': [
                            '620 - Government Buildings (Includes Courthouse, Post Office',
                            '625 - Garbage Dumps, Sanitary Fills, Sewer Lagoons, Etc.',
                            '630 - Works Yards',
                            '634 - Government Research Centres (Includes Nurseries &',
                            '640 - Hospitals (Nursing Homes Refer To Commercial Section).',
                            '642 - Cemeteries (Includes Public Or Private).',
                            '650 - Schools & Universities, College Or Technical Schools',
                            '652 - Churches & Bible Schools',
                        ],
                    }

                    """
                            'vacant': ['001 - Vacant Residential Less Than 2 Acres', '051 - Multi-Family (Vacant)',
                                       '061 - 2 Acres Or More (Vacant)',
                                       '111 - Grain & Forage (Vacant)',
                                       '121 - Vegetable & Truck (Vacant)',
                                       '201 - Vacant IC&I',
                                       '421 - Vacant', '422 - IC&I Water Lot (Vacant)',
                                       '601 - Civic, Institutional & Recreational (Vacant)'],
                            'parking': ['020 - Residential Outbuilding Only', '029 - Strata Lot (Parking Residential)',
                                        '043 - Parking (Lot Only, Paved Or Gravel-Res)',
                                        '219 - Strata Lot (Parking Commercial)',
                                        '260 - Parking (Lot Only, Paved Or Gravel-Com)', '262 - Parking Garage',
                                        '490 - Parking Lot Only (Paved Or Gravel)'],
                            'other': ['002 - Property Subject To Section 19(8)', '070 - 2 Acres Or More (Outbuilding)',
                                      '190 - Other',
                                      '200 - Store(S) And Service Commercial', '205 - Big Box',
                                      '220 - Automobile Dealership', '222 - Service Station',
                                      '224 - Self-Serve Service Station',
                                      '226 - Car Wash', '227 - Automobile Sales (Lot)',
                                      '228 - Automobile Paint Shop, Garages, Etc.',
                                      '258 - Drive-In Restaurant', '288 - Sign Or Billboard Only',
                                      '590 - Miscellaneous (Transportation & Communication)',
                                      '615 - Government Reserves',
                                      '632 - Ranger Station'],
                            'agriculture': ['110 - Grain & Forage', '111 - Grain & Forage',
                                            '120 - Vegetable & Truck', '130 - Tree Fruits',
                                            '140 - Small Fruits', '150 - Beef', '160 - Dairy', '170 - Poultry',
                                            '180 - Mixed',
                                            '240 - Greenhouses And Nurseries (Not Farm Class)'],
                            'industrial': ['031 - Strata-Lot Self Storage-Res Use',
                                           '217 - Air Space Title',
                                           '218 - Strata-Lot Self Storage-Business Use',
                                           '272 - Storage & Warehousing (Open)',
                                           '273 - Storage & Warehousing (Closed)', '274 - Storage & Warehousing (Cold)',
                                           '275 - Self Storage', '276 - Lumber Yard Or Building Supplies',
                                           '300 - Stratified Operational Facility Areas',
                                           '400 - Fruit & Vegetable',
                                           '401 - Industrial (Vacant)', '402 - Meat & Poultry', '403 - Sea Food',
                                           '404 - Dairy Products', '405 - Bakery & Biscuit Manufacturing',
                                           '406 - Confectionery Manufacturing & Sugar Processing', '408 - Brewery',
                                           '409 - Winery',
                                           '414 - Miscellaneous (Food Processing)',
                                           '415 - Sawmills',
                                           '416 - Planer Mills (When Separate From Sawmill)',
                                           '418 - Shingle Mills',
                                           '419 - Sash & Door',
                                           '420 - Lumber Remanufacturing (When Separate From Sawmill)',
                                           '423 - IC&I Water Lot (Improved)',
                                           '424 - Pulp & Paper Mills (Incl Fine Paper, Tissue & Asphalt Roof)',
                                           '425 - Paper Box, Paper Bag, And Other Paper Remanufacturing.',
                                           '426 - Logging Operations, Incl Log Storage',
                                           '428 - Improved',
                                           '429 - Miscellaneous (Forest And Allied Industry)',
                                           '434 - Petroleum Bulk Plants',
                                           '445 - Sand & Gravel (Vacant and Improved)',
                                           '447 - Asphalt Plants',
                                           '448 - Concrete Mixing Plants',
                                           '449 - Miscellaneous (Mining And Allied Industries)',
                                           '452 - Leather Industry',
                                           '454 - Textiles & Knitting Mills', '456 - Clothing Industry',
                                           '458 - Furniture & Fixtures Industry',
                                           '460 - Printing & Publishing Industry',
                                           '462 - Primary Metal Industries (Iron & Steel Mills,',
                                           '464 - Metal Fabricating Industries',
                                           '466 - Machinery Manufacturing (Excluding Electrical)',
                                           '470 - Electrical & Electronics Products Industry',
                                           '472 - Chemical & Chemical Products Industries',
                                           '474 - Miscellaneous & (Industrial Other)',
                                           '476 - Grain Elevators',
                                           '478 - Docks & Wharves',
                                           '480 - Shipyards',
                                           '500 - Railway',
                                           '505 - Marine & Navigational Facilities (Includes Ferry',
                                           '510 - Bus Company, Including Street Railway', '520 - Telephone',
                                           '530 - Telecommunications (Other Than Telephone)',
                                           '515 - Airports, Heliports, Etc.',
                                           '540 - Community Antenna Television (Cablevision)',
                                           '550 - Gas Distribution Systems',
                                           '560 - Water Distribution Systems',
                                           '580 - Electrical Power Systems (Including Non-Utility']
                    }

                    # Broaden 'other' uses for a better accuracy on diversity indexes
                    uses['other'] = uses['other'] + uses['vacant'] + uses['parking'] + uses['agriculture'] + uses['industrial']
                    uses['vacant'], uses['parking'], uses['agriculture'], uses['industrial'] = [], [], [], []
"""

                    new_uses = []
                    index = list(out_gdf.columns).index("PRIMARY_ACTUAL_USE")
                    all_prim_uses = [item for sublist in list(elab_uses.values()) for item in sublist]
                    for row in out_gdf.iterrows():
                        for key, value in elab_uses.items():
                            if row[1]['PRIMARY_ACTUAL_USE'] in value:
                                new_uses.append(key)
                        if row[1]['PRIMARY_ACTUAL_USE'] not in all_prim_uses:
                            new_uses.append('other')
                    out_gdf['n_use'] = new_uses

                    # Export assessment fabric layer to GeoPackage
                    out_gdf.to_file(f"{city.directory}Databases/BCAssessment/{city.city_name}_fabric.geojson", driver='GeoJSON')
                    out_gdf.to_file(city.gpkg, driver='GPKG', layer='land_assessment_fabric')

                    # Delete repeated parcels
                    p_gdf = out_gdf.drop_duplicates(subset=['geometry'])

                    # Classify parcels into categories
                    p_gdf.loc[:, 'area'] = p_gdf.loc[:, 'geometry'].area
                    p_gdf.loc[p_gdf['area'] < 400, 'n_size'] = 'less than 400'
                    p_gdf.loc[(p_gdf['area'] > 400) & (p_gdf['area'] < 800), 'n_size'] = '400 to 800'
                    p_gdf.loc[(p_gdf['area'] > 800) & (p_gdf['area'] < 1600), 'n_size'] = '800 to 1600'
                    p_gdf.loc[(p_gdf['area'] > 1600) & (p_gdf['area'] < 3200), 'n_size'] = '1600 to 3200'
                    p_gdf.loc[(p_gdf['area'] > 3200) & (p_gdf['area'] < 6400), 'n_size'] = '3200 to 6400'
                    p_gdf.loc[p_gdf['area'] > 6400, 'n_size'] = 'more than 6400'

                    # Convert area to square km
                    p_gdf.loc[:, 'area_sqkm'] = p_gdf.loc[:, 'geometry'].area/100000

                    # Export parcel layer to GeoPackage
                    p_gdf = p_gdf.drop('area', axis=1)
                    p_gdf['geometry'] = p_gdf['geometry'].buffer(0)

                    # Export geojson and read again to try to export to GeoPackage
                    p_gdf.to_file(f"{city.directory}Databases/BCAssessment/{city.city_name}_parcels.geojson", driver='GeoJSON')
                    p_gdf.to_file(city.gpkg, driver='GPKG', layer='land_assessment_parcels')
                    elapsed = round((timeit.default_timer() - start_time) / 60, 1)
                    return print(f"> Data aggregation from BC Assessment finished in {elapsed} minutes")

    # BC Transit
    def get_bc_transit(self, urls, run=True, down=True):
        if run:
            if len(urls) != len(self.cities): print("Number of urls and BC cities must be the same")
            else:
                for city, url in zip(self.cities, urls):
                    dir = f'{city.directory}/BCTransit'
                    if down:
                        print(f"Downloading transit data from BC Transit for {city.municipality}")
                        zip_path = f"{dir}/{city.municipality}.{url.split('/')[len(url.split('/'))-1]}"
                        filename = download_file(url, file_path=zip_path)
                        os.rename(filename, zip_path)

                    # Create directory if it doesn't exist
                    n_dir = f"{dir}/{city.municipality}"
                    if os.path.exists(n_dir): pass
                    else: os.mkdir(n_dir)

                    if down:
                        # Unzip data and delete file
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extractall(n_dir)
                        os.remove(zip_path)

                    # Read stops and stop_times
                    stops = pd.read_csv(f"{n_dir}/stops.txt")
                    stops.set_index('stop_id')
                    times = pd.read_csv(f"{n_dir}/stop_times.txt")
                    calen = pd.read_csv(f"{n_dir}/calendar_dates.txt")

                    # Calculate frequency of trips on each stop
                    for stop in times['stop_id'].unique():
                        s_df = times.loc[times['stop_id'] == stop]
                        n_trips = len(s_df['trip_id'].unique())
                        frequency = n_trips/len(calen)
                        stops.loc[stops['stop_id'] == stop, 'n_trips'] = int(n_trips)
                        stops.loc[stops['stop_id'] == stop, 'frequency'] = frequency
                        print(f"> Stop {stop} processed with {n_trips} trips at a frequency of {frequency} trips per day")

                    # Create stop geometries
                    stops['geometry'] = [Point(lon, lat) for lon, lat in zip(stops.stop_lon, stops.stop_lat)]
                    stops['z_id'] = stops['zone_id']
                    stops = stops.drop('zone_id', axis=1)
                    stops_gdf = gpd.GeoDataFrame(stops, geometry='geometry')
                    stops_gdf = stops_gdf.dropna(subset=['frequency', 'geometry'], axis=0)[stops_gdf.geometry.is_valid]
                    stops_gdf = stops_gdf.reset_index(drop=True)

                    # Reproject and export
                    stops_gdf.crs = 4326
                    stops_gdf = stops_gdf.to_crs(city.crs)
                    stops_gdf.to_file(city.gpkg, layer='network_stops', driver='GPKG')
                    print(f"Transit stops for {city.municipality} saved at 'network_stops' layer")

                    # Get route lengths
                    routes = pd.read_csv(f"{n_dir}/routes.txt")
                    shapes = pd.read_csv(f"{n_dir}/shapes.txt")
                    lines = {'route': [], 'geometry': []}
                    all_pts = []
                    for i in shapes['shape_id'].unique():
                        pts = []
                        print(f'> Processing route {i}')
                        for j in shapes[shapes['shape_id'] == i].sort_values('shape_pt_sequence')['shape_pt_sequence']:
                            stop = shapes[(shapes['shape_id'] == i) & (shapes['shape_pt_sequence'] == j)]
                            try:
                                pts.append(Point(stop['shape_pt_lon'].values[0], stop['shape_pt_lat'].values[0]))
                                all_pts.append(Point(stop['shape_pt_lon'].values[0], stop['shape_pt_lat'].values[0]))
                            except: pass

                        if len(pts) > 0:
                            lines['route'].append(i)
                            lines['geometry'].append(LineString(pts))

                    lines = gpd.GeoDataFrame(lines, geometry='geometry')
                    lines.crs = 4326
                    lines = lines.to_crs(26910)
                    lines['length'] = [geom.length for geom in lines['geometry']]
                    lines.to_file(city.gpkg, layer='network_routes', driver='GPKG')
                    print(f"Transit routes for {city.municipality} saved at 'network_routes' layer")
        return

    def get_air_quality(self):
        url = "http://www.env.gov.bc.ca/epd/bcairquality/aqo/csv/bc_air_monitoring_stations.csv"
        fp = download_file(url=url, file_path=f"{self.cities[0].directory}/OpenDataBC/{url.split('/')[-1]}")

        df = pd.read_csv(fp)

        for city in self.cities:
            # Load and reproject data
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE))
            gdf.crs = 4326
            gdf = gdf.to_crs(city.crs)

            # Reproject city boundary
            boundary = gpd.read_file(city.gpkg, layer='land_municipal_boundary')
            bound = boundary.to_crs(city.crs)

            # Calculate Air Quality Index
            gdf['aqi'] = 1000 / (gdf['PM25'] + gdf['NO2'] + gdf['O3']) # Source: https://en.wikipedia.org/wiki/Air_Quality_Health_Index_(Canada)

            # Join data within bounds
            try: gdf = gpd.overlay(gdf, bound, how='intersection')
            except: pass
            if len(gdf) > 0:
                gdf.to_file(city.gpkg, layer='bc_air_quality', driver='GPKG')
                print(f"Air quality data downloaded with {len(gdf)} features for {city.municipality}")
            else: print(f"Air quality not downloaded for {city.municipality}")
