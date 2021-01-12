# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:52:23 2019

@author: nicholas-martino
"""

import datetime

from Scraper import BritishColumbia
from _Settings import regions

print(f"Start @ {datetime.datetime.now()}")

for key, value in regions.items():
    bc = BritishColumbia(cities=value['British Columbia'])

    # Set parameters
    radius = [4800, 3200, 1600, 800, 400]
    filter_min = {'population density per square kilometre, 2016': 300}
    decays = ['flat', 'linear']

    # Iterate over cities (settings)
    for city in bc.cities:

        # Interpolate dependent variables
        city.spatial_interpolation(run=False, layer='bc_air_quality', feature='aqi', resolution=100)
        city.spatial_join(run=False, sample_layer='bc_air_quality_interp', aggregated_layers={
            'land_health_regions': ["covid_case"]})

        # Scrape and calculate network measures
        city.node_elevation(run=False)
        city.centrality(run=False, axial=True, layer='network_walk')
        city.centrality(run=False, osm=True, layer='network_drive')

        # Run network analysis to aggregate dependent variables
        run_net_dep = False
        city.directory = f"{city.directory}/Network"
        city.network_analysis(run=run_net_dep,
            prefix='den', feature_layer='network_analysis_livability', filter_min=filter_min, decays=['flat', 'linear'],
            service_areas=radius, sample_layer='land_dissemination_area',
            aggregated_layers={
                'land_dissemination_area': ["population density per square kilometre, 2016"],
            })
        city.network_analysis(run=run_net_dep,
            prefix='acc', feature_layer='network_analysis_livability', filter_min=filter_min, decays=decays,
            service_areas=[400], sample_layer='land_dissemination_area',
            aggregated_layers={
                'network_stops': ["frequency"],
                'land_dissemination_area': ["walk", "bike"]
            })
        city.network_analysis(run=run_net_dep,
            prefix='div', feature_layer='network_analysis_livability', filter_min=filter_min, decays=decays,
            service_areas=[400], sample_layer='land_dissemination_area',
            aggregated_layers={
                'land_dissemination_area': ["educ_div_sh", "income_div_sh", "age_div_sh", "ethnic_div_sh"],
            })
        city.network_analysis(run=run_net_dep,
            prefix='aff', feature_layer='network_analysis_livability', filter_min=filter_min, decays=decays,
            service_areas=[400], sample_layer='land_dissemination_area',
            aggregated_layers={
                'craigslist_rent': ["price_sqft"],
                'land_dissemination_area': ["owned_ave_cost", "owned_ave_dwe_value", "more30%income_rat"]
            })
        city.network_analysis(run=run_net_dep,
            prefix='vit', feature_layer='network_analysis_livability', filter_min=filter_min, decays=decays,
            service_areas=[400], sample_layer='land_dissemination_area',
            aggregated_layers={
                'indeed_employment': [],
                'land_dissemination_area': ["employment rate"],
                'land_assessment_fabric': ["n_use"]
            })
        city.network_analysis(run=run_net_dep,
            prefix='hth', feature_layer='network_analysis_livability', filter_min=filter_min, decays=decays,
            service_areas=[400], sample_layer='land_dissemination_area',
            aggregated_layers={
                'bc_air_quality_interp_': ["aqi", "covid_case_mean"]
            })
        city.network_analysis(run=run_net_dep,
            prefix='sft', feature_layer='network_analysis_livability', filter_min=filter_min, decays=decays,
            service_areas=[400], sample_layer='land_dissemination_area',
            aggregated_layers={
                'icbc_accidents': ["crash count"],
            })

        # Run network analysis to aggregate independent variables
        run_net_ind = False
        city.network_analysis(run=run_net_ind,
            prefix='int', feature_layer='network_analysis_livability', filter_min=filter_min, decays=decays,
            service_areas=radius, sample_layer='land_dissemination_area',
            aggregated_layers={
                'network_drive': ["length"],
                'land_ecohealth': ["cc_per", "imp_per", "pcc_veg"],
                'land_assessment_fabric': ["year_built", "total_finished_area", "number_of_storeys",
                    "number_of_bedrooms", "number_of_bathrooms", "gross_building_area"],
                'land_assessment_parcels': ["area"],
                'land_dissemination_area': ["n_dwellings", "total_bedr", "ave_n_rooms"]
            })
        city.network_analysis(run=run_net_ind,
            prefix='mix', feature_layer='network_analysis_livability', filter_min=filter_min, decays=decays,
            service_areas=radius, sample_layer='land_dissemination_area',
            aggregated_layers={
                'land_assessment_parcels': ["n_size"],
                'land_dissemination_area': ["dwelling_div_rooms_si", "dwelling_div_bedrooms_si",
                    "dwelling_div_rooms_sh", "dwelling_div_bedrooms_sh", "building_age_div_si", "building_age_div_sh"]
            })
        city.network_analysis(run=True,
            prefix='ctr', feature_layer='network_analysis_livability', filter_min=filter_min, decays=decays,
            service_areas=radius, sample_layer='land_dissemination_area',
            aggregated_layers={
                'network_nodes': [],  # ["node_closeness", "node_betweenness"],
                'network_drive': ["length"],  # , "link_betweenness", "link_n_betweenness"],
                'network_axial': ["axial_length", "connectivity", "axial_closeness", "axial_betweenness",
                    "axial_n_betweenness", "axial_degree", "axial_pagerank", "axial_eigenvector", "axial_katz",
                    "axial_hits1"],
            })

print(f"Finished @ {datetime.datetime.now()}")
