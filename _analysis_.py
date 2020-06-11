# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:52:23 2019

@author: nicholas-martino
"""

import time
import datetime
from _settings_ import regions
from Scraper import BritishColumbia


print(f"Start @ {datetime.datetime.now()}")
for i in range(100000):

    for key, value in regions.items():
        bc = BritishColumbia(cities=value['British Columbia'])

        # Analyze data
        radius = [400, 800, 1600, 3200, 6400]
        for city in bc.cities:
            city.node_elevation(run=False)
            city.centrality(run=False, axial=True, layer='network_links')
            city.centrality(run=False, osm=True, layer='network_drive')
            city.network_analysis(run=False,
                prefix='mph', feature_layer='network_analysis_livability',
                service_areas=radius, decays=['flat', 'linear'], sample_layer='land_dissemination_area',
                aggregated_layers={
                    'network_axial': [
                        "axial_length", "connectivity", "axial_closeness", "axial_betweenness", "axial_n_betweenness"],
                    'network_drive': ["length", "link_betweenness", "link_n_betweenness"],
                    'network_nodes': ["node_closeness", "node_betweenness"],
                    'land_assessment_fabric': ["total_finished_area", "number_of_storeys", "number_of_bedrooms",
                        "number_of_bathrooms"],
                    'land_assessment_parcels': ["area", "n_size"],
                    'land_dissemination_area': ["n_dwellings", "total_bedr", "ave_n_rooms",
                        "dwelling_div_rooms_si", "dwelling_div_bedrooms_si", "dwelling_div_rooms_sh",
                        "dwelling_div_bedrooms_sh", "building_age_div_si", "building_age_div_sh"]
                })
            city.spatial_join(run=True, prefix='den', sample_layer='land_dissemination_area',
                aggregated_layers={
                    'land_dissemination_area': ["Population density per square kilometre, 2016",
                        "1_person_per_room_ratio", "1_plus_person_per_room_ratio"]
                })
            city.spatial_join(run=True, prefix='acc_div', sample_layer='land_dissemination_area',
                aggregated_layers={
                    'network_stops': ["frequency"],
                    'land_assessment_fabric': ["n_use_diversity_sh", "n_use_diversity_si"],
                    'land_dissemination_area': ["diversity_educ_sh", "diversity_educ_si",
                        "income_div_sh", "income_div_si"]
                })
            city.spatial_join(run=True, prefix='hth_sft', sample_layer='land_dissemination_area',
                aggregated_layers={
                    'gps_traces': ["time"],
                    'air_quality': ["aqi"],
                    'icbc_accidents': ["Crash Count"]
                })
            city.spatial_join(run=True, prefix='aff_vit', sample_layer='land_dissemination_area',
                aggregated_layers={
                    'craigslist_housing': ["price_sqft"],
                    'employment_indeed': ["salary"],
                    'land_assessment_fabric': ["retail_use", "office_use", "entertainment_use", "civic_use"],
                    'land_dissemination_area': ["owner_ratio", "median_income", "without_income_at",
                        "more30%income_rat", "owned_ave_cost", "owned_med_cost", "owned_ave_dwe_value",
                        "owned_med_dwe_value", "Unemployment rate", "income_div_sh", "income_div_si"]
                })

    print(f"Finished @ {datetime.datetime.now()}")
