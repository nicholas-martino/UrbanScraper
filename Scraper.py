# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:52:23 2019

@author: nicholas-martino (C) MIT License

"""

import datetime
import glob
import io
import json
import logging
import os
import time
from io import StringIO
from urllib import request, error, parse
from elementslab.Analyst import GeoBoundary
import geopandas as gpd
import osmnx as ox
import pandas as pd
import regex
import requests
import selenium.webdriver as webdriver
from Learning.Scraping import Scraper
from Statistics.basic_stats import normalize
from Visualization import polar_chart as pc
from bs4 import BeautifulSoup
from craigslist import CraigslistHousing
from geopy.geocoders import Nominatim
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from selenium.webdriver.firefox.options import Options
from shapely import wkt
from shapely.geometry import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


class GeoScraper:
    """Web scraping GeoSpatial data"""
    def __init__(self, city):
        self.city = city

    # Scrape built environment
    def buildings_osm(self):
        return self

    # Scrape affordability and vitality indicators
    def employment_craigslist(self):
        return self

    def employment_indeed(self, n_pages=3):
        print(f"Downloading employment posts from Indeed")

        # Build search
        name = self.city.municipality.replace(',', '%2C').replace(' ', '+')
        start = 0

        # Create selenium driver
        s = Scraper()
        s.driver.minimize_window()

        data = {'job': [], 'time': [], 'salary': [], 'company': [], 'geometry': []}
        for i in range(start, n_pages):
            url = f"https://www.indeed.ca/jobs?q=&l={name}&start={i*20}"

            # Access and parse the page
            page = requests.get(url)
            soup = BeautifulSoup(page.text, "html.parser")

            # Extract job titles
            for div in soup.find_all(name="div", attrs={"class": "row"}):
                for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
                    data['job'].append(a["title"])

                    # Extract time stamps
                    page1 = requests.get(f"https://www.indeed.ca/{a['href']}")
                    soup1 = BeautifulSoup(page1.text, "html.parser")
                    for div1 in soup1.find_all(name="div", attrs={"class": "jobsearch-JobMetadataFooter"}):
                        p_days = div1.find_all(text=True)[1]
                        for j in p_days.split():
                            try:
                                result = float(j)
                                break
                            except: result = 0
                        data['time'].append(f"{datetime.datetime.now().date()-datetime.timedelta(days=result)} - {datetime.datetime.now()}")

            # Extract salaries
            for div in soup.find_all(name="div", attrs={"class": "row"}):
                if len(div.find_all(name="span", attrs={"class": "salaryText"})) == 0: data['salary'].append('Unknown')
                else:
                    for sal in div.find_all(name="span", attrs={"class": "salaryText"}):
                        try: data['salary'].append(sal.text)
                        except:
                            try: data['salary'].append(sal[0].text)
                            except: data['salary'].append('Unknown')

            # Extract companies
            for div in soup.find_all(name="div", attrs={"class": "row"}):
                company = div.find_all(name="span", attrs={"class": "company"})
                if len(company) > 0:
                    for b in company:
                        data['company'].append(b.text.strip())
                else:
                    sec_try = div.find_all(name="span", attrs={"class": "result - link - source"})
                    for span in sec_try:
                        data['company'].append(span.text.strip())

            # Clean street names from OSM
            cl_strs = [i.replace("Avenue", "").replace("Street", "").replace("Road", "").replace("Drive", "").strip(" ")
                       for i in list(self.city.links.name.unique()) if i is not None]

            # Match locations from job names
            rec_locs = []
            for job in data['job']:
                locs = []
                for word in word_tokenize(job):
                    if word in cl_strs: locs.append(word)
                rec_locs.append(locs)
            rec_locs = ['+'.join(l) for l in rec_locs]

            # Extract general location
            locations = []
            divs = soup.findAll("div", attrs={"class": "recJobLoc"})
            for div in divs:
                locations.append(div['data-rc-loc'])

            # Get urls to search Google Maps
            urls = [f"https://www.google.com/maps/place/{c.replace(' ', '+')}+{l.replace(' ', '+')}+{rl}"
                    for c, l, rl in zip(data['company'], locations, rec_locs)]

            # Extract point location from Google Map based on company names
            for url in urls:

                # Load URL
                s.driver.get(url)
                time.sleep(2)

                # Press search button
                try: s.driver.find_elements_by_id("searchbox-searchbutton")[0].click()
                except: s.driver.find_elements_by_id("searchbox-searchbutton").click()
                time.sleep(2)

                # Get address of first item
                try: address = s.driver.find_elements_by_class_name("section-result-location")[0].text
                except:
                    try: address = s.driver.find_elements_by_class_name("section-info-text")[0].text
                    except: address = None

                # Get point location from address
                locator = Nominatim(user_agent="myGeocoder")
                try: geom = Point(locator.geocode(f"{address}, {self.city.municipality}")[1])
                except:
                    try: geom = Point(locator.geocode(f"{address}, {self.city.city_name}")[1])
                    except: geom = 'Unknown'
                data['geometry'].append(geom)

        # Close web browser
        s.driver.close()

        # Export to GeoPackage
        gdf = gpd.GeoDataFrame().from_dict(data)
        gdf = gdf.loc[gdf['geometry'] != 'Unknown']
        gdf.geometry = [Point(t.y, t.x) for t in gdf.geometry]
        gdf.crs = 4326
        gdf.to_crs(epsg=self.city.crs, inplace=True)
        self.city.boundary.to_crs(epsg=self.city.crs, inplace=True)
        try:
            gdf0 = gpd.read_file(self.city.gpkg, layer='indeed_employment')
            gdf = pd.concat([gdf0, gdf])
        except: pass
        gdf.drop_duplicates(inplace=True)
        gdf = gpd.overlay(gdf, self.city.boundary)
        gdf.to_file(self.city.gpkg, layer='indeed_employment')
        print(f"> Employment data downloaded from Indeed")
        return self

    def housing_craigslist(self, site, n_results):
        print(f"Downloading {self.city.city_name}'s housing posts from Craigslist")
        cl = CraigslistHousing(site=site)
        results = cl.get_results(sort_by='newest', geotagged=True, limit=n_results)

        # List results
        uid = []
        name = []
        area = []
        price = []
        brooms = []
        coords = []
        dates = []
        for result in results:
            uid.append(result['id'])
            name.append(result['name'])
            area.append(result['area'])
            brooms.append(result['bedrooms'])
            coords.append(result['geotag'])
            price.append(float(result[('price')][1:]))
            dates.append(result['datetime'])
            # print (result)

        # Format coordinates
        coord_x = []
        coord_y = []
        for coord in coords:
            split = str(coord).split(',')
            try:
                coord_x.append(str(split[1])[1:][:-1])
                coord_y.append(str(split[0])[1:][:-1])
            except:
                coord_x.append(str(0.00))
                coord_y.append(str(0.00))
        a = []
        for i in coord_x: a.append(f'POINT ({str(i)}')
        b = []
        for i in coord_y: b.append(' ' + str(i) + ')')

        # Remove null items
        ccoord = ([str(x + y) for x, y in zip(a, b)])
        df = pd.DataFrame(
            {'id': uid, 'name': name, 'price': price, 'area': area, 'bedrooms': brooms, 'geometry': ccoord,
             'date': dates})
        coord_nnull = df['geometry'] != "POINT (0.0 0.0)"
        area_nnull = df['area'].notnull()
        df = df[(area_nnull) & (coord_nnull)]

        # Geoprocess Coordinates
        df['geometry'] = df['geometry'].apply(wkt.loads)
        cl_gdf = gpd.GeoDataFrame(df, geometry='geometry')
        cl_gdf.crs = 4326

        # Calculate Price/Area
        cl_gdf['area'] = cl_gdf['area'].str[:-3].astype(float)
        cl_gdf['price_sqft'] = cl_gdf['price'] / cl_gdf['area']

        # Get Vancouver Boundary
        van_gdf = ox.gdf_from_place(self.city.municipality)
        van_gdf.crs = 4326
        van_gdf.to_crs(epsg=self.city.crs)
        van_pol = van_gdf.geometry[0]
        if van_pol is None: print('Administrative boundary download failed :(')

        # Filter data
        pd.set_option('display.min_rows', 10)
        cl_gdf.to_crs(epsg=self.city.crs)
        cl_gdf = cl_gdf[cl_gdf.within(van_pol)]
        csv_path = '__pycache__/' + self.city.municipality + '_Craigslist.csv'
        cl_gdf = cl_gdf[cl_gdf['area'] > 270]

        # Write data in csv
        with io.open(csv_path, "a", encoding="utf-8") as f:
            cl_gdf.to_csv(f, header=False, index=False)
        try:
            df = pd.read_csv(csv_path)
            df = df.drop_duplicates()
            df.to_csv(csv_path, header=False, index=False)

            # Write data in GeoPackage
            df = pd.read_csv(csv_path, encoding="utf8")
            df.columns = ['index', 'description', 'price', 'area', 'bedrooms', 'geometry', 'date', 'price_sqft']
            numerics = ['price', 'area', 'bedrooms', 'price_sqft']
            for i in numerics: pd.to_numeric(df[i], errors='coerce').fillna(0).astype(float)
            gdf = gpd.GeoDataFrame(df)
            gdf['geometry'] = gdf['geometry'].apply(wkt.loads)
            gdf.set_geometry('geometry')
            gdf.crs = 4326
            try:
                cl_cur = gpd.read_file(self.city.gpkg, layer='craigslist_housing', driver='GPKG')
                try: cl_cur.drop('fid', inplace=True)
                except: pass
                gdf = pd.concat([cl_cur, gdf])
                gdf.drop_duplicates(inplace=True)
                print(f"Craigslist data for {self.city.city_name} downloaded and joined")
            except: pass
            gdf.to_file('Databases/' + self.city.municipality + '.gpkg', layer='craigslist_housing', driver="GPKG")
            print(f"Craigslist data for {self.city.city_name} exported to GeoPackage")

        except:
            gdf = None
            print(f"Failed to process craigslist data for {self.city.city_name} :(")
            pass
        return gdf

    def housing_zillow(self, url='https://www.zillow.com/homes/british-columbia_rb/'):

        options = Options()
        options.set_preference("browser.download.folderList", 1)
        options.set_preference("browser.download.manager.showWhenStarting", False)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")
        wd = webdriver.Firefox(executable_path=r'C:\WebDrivers\geckodriver.exe', options=options)

        try:
            list = self.city_name.split(' ')
            name = list[0] + '-' + list[1]
        except:
            name = self.city_name

        url = 'https://www.zillow.com/homes/' + name + '_rb/'
        wd.get(url)
        wd.close()

        """
        r = urllib.request.urlopen(base_url)
        parser = html.fromstring(r.text)

        print(r)
        print(r.text)

        response = requests.get(base_url)
        print("status code received:", response.status_code)
        print(response.text)
        print(response.content)

        parser = html.fromstring(response.text)
        print(str(parser.text))
        search_results = parser.xpath("//div[@id='search-results']//article")
        print(search_results)

        z = Zillow(name=self.municipality, base_url=base_url)
        data = z.run()

        zillow_data = ZillowWrapper(zillow_id)
        updated_property_details_response = zillow_data.get_updated_property_details(zillow_id)
        result = GetUpdatedPropertyDetails(updated_property_details_response)
        print(result.rooms)  # number of rooms of the home
        """
        return None

    def bc_assessment(self):
        print(self.city_name)
        # Get caprate data via webscraping bc assessment webpage
        # Get and calculate land value data from tax property assessment and selling/renting
        return None

    # Scrape health and safety indicators
    def air_quality(self, token):
        """
        A Python wrapper for AQICN API.
        The library can be used to search and retrieve Air Quality Index data.
        Please refer to AQICN website to obtain token that must be used for access.
        """

        endpoint = 'https://api.waqi.info/'
        print(f"Downloading air quality data from {endpoint}")
        endpoint_search = endpoint + 'search/'
        endpoint_obs = endpoint + 'feed/@%d/'
        endpoint_geo = endpoint + 'feed/geo:%d;%d/'

        def find_station_by_city(city_name, token):
            """Lookup AQI database for station codes in a given city."""
            req = requests.get(
                endpoint_search,
                params={
                    'token': token,
                    'keyword': city_name
                })

            if req.status_code == 200 and req.json()["status"] == "ok":
                return [result["uid"] for result in req.json()["data"]]
            else:
                return []

        def get_location_observation(lat, lng, token):
            """Lookup observations by geo coordinates."""
            req = requests.get(
                endpoint_geo % (lat, lng),
                params={
                    'token': token
                })

            if req.status_code == 200 and req.json()["status"] == "ok":
                return parse_observation_response(req.json()["data"])
            return {}

        def parse_observation_response(json):
            """Decode AQICN observation response JSON into python object."""
            logging.debug(json)

            try:
                iaqi = json['iaqi']
                result = {
                    'idx': json['idx'],
                    'city': json.get('city', ''),
                    'aqi': json['aqi'],
                    'dominentpol': json.get("dominentpol", ''),
                    'time': json['time']['s'],
                    'iaqi': [{'p': item, 'v': iaqi[item]['v']} for item in iaqi]
                }
                return result
            except:
                print(f'No air quality index data found for station {station}')

        def get_station_observation(station_code, token):
            """Request station data for a specific station identified by code.
            A language parameter can also be specified to translate location
            information (default: "en")
            """
            req = requests.get(
                endpoint_obs % (station_code),
                params={
                    'token': token
                })

            if req.status_code == 200 and req.json()['status'] == "ok":
                return parse_observation_response(req.json()['data'])
            else:
                return {}

        stations = find_station_by_city(self.city.municipality, token)
        observations = {}
        for station in stations:
            observations[station] = get_station_observation(station, token)

        # Create function to extract data from dict
        get_val = lambda col: [observations[key][col] for key in list(observations.keys()) if observations[key] is not None]

        # Create DataFrame and geo reference it
        df = pd.DataFrame.from_dict({'lat': [d['geo'][0] for d in get_val('city')], 'long': [d['geo'][1] for d in get_val('city')]})
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.long, df.lat))

        # Get data from dict into GeoDataFrame
        gdf['idx'] = get_val('idx')
        gdf['aqi'] = get_val('aqi')
        gdf.replace("-", 0, inplace=True)
        gdf['aqi'].astype(float)
        gdf['time'] = get_val('time')
        gdf['dominentpol'] = get_val('dominentpol')

        # Reproject data
        gdf.crs = 4326
        gdf.to_crs(self.city.crs)

        # Check if GeoDataFrame exists and append new data if it does
        try:
            aq_cur = gpd.read_file(self.city.gpkg, layer='air_quality', driver='GPKG')
            try: aq_cur.drop('fid', inplace=True)
            except: pass
            gdf = pd.concat([aq_cur, gdf])
        except: pass

        # Export to GeoPackage
        gdf.to_file(self.city.gpkg, layer='air_quality', driver='GPKG')
        return gdf

    def landscape_greenness(self):
        return self

    def landscape_trees(self):
        # Get existing trees from satellite and street view images
        return print('done')

    # Scrape accessibility and social diversity indicators
    def movement_osm_gps(self, run=True):
        if run:
            print(f"Downloading {self.city.municipality}'s public GPS traces from OpenStreetMaps")
            area = self.city.boundary.geometry.to_crs(epsg=4326).geometry.area[0]
            boundaries = [self.city.boundary.to_crs(epsg=4326).geometry[0]]
            directory = f"{self.city.directory}Databases/osm/{self.city.municipality}"
            if os.path.exists(directory): pass
            else: os.mkdir(directory)

            while area > 0.25:
                out_bounds = []
                # Break each boundary into four parts
                for b in boundaries:
                    cutter = MultiLineString([
                        LineString([(b.bounds[0]-0.1, b.centroid.y), (b.bounds[2]+0.1, b.centroid.y)]),
                        LineString([(b.centroid.x, b.bounds[1]-0.1), (b.centroid.x, b.bounds[3]+0.1)])
                    ]).buffer(0.000001)
                    boundaries = b.difference(cutter)
                    out_bounds = out_bounds + [b for b in boundaries]
                # Check area of the bounding box of each splitted part (out_bounds)
                bbox_areas = [Polygon([
                    Point(b.bounds[0], b.bounds[1]),
                    Point(b.bounds[0], b.bounds[3]),
                    Point(b.bounds[2], b.bounds[3]),
                    Point(b.bounds[2], b.bounds[1])
                ]).area for b in out_bounds]

                area = max(bbox_areas)
                if area < 0.25:
                    boundaries = out_bounds
                    print(f"Area too big, dividing boundary into {len(out_bounds)} polygons")
                    break

            try: gdf = gpd.read_file(self.city.gpkg, layer='gps_traces')
            except: gdf = gpd.GeoDataFrame(columns=['file'])

            for i, b in enumerate(boundaries):
                min_ln, min_lt, max_ln, max_lt = b.bounds

                # Verify existing files
                try: init = int(max([float(k[2:-3]) for k in os.listdir(directory) if f"{i}_" in k])) + 1
                except: init = 0

                # Start downloading
                for j in range(init, 100000):
                    page = j
                    url = f"http://api.openstreetmap.org/api/0.6/trackpoints?bbox={min_ln},{min_lt},{max_ln},{max_lt}&page={page}"
                    try:
                        file_name = f"{i}_{page}.gpx"
                        print(f"Saving trace from {url} to {directory}/{file_name}")

                        u = request.urlopen(url)
                        buffer = u.read()

                        if len(buffer) < 300:
                            print(f"File size of {len(buffer)} is too small, jumping to next space")
                            break
                        else:
                            f = open(f"{directory}/{file_name}", 'wb')
                            f.write(buffer)
                            f.close()
                            time.sleep(1)
                            if file_name not in gdf['file'].unique():
                                gdf1 = gpd.read_file(f"{directory}/{file_name}", layer='track_points')
                                gdf1['file'] = file_name
                                gdf1.crs = 4326
                                gdf1.to_crs(epsg=self.city.crs, inplace=True)
                                gdf = pd.concat([gdf, gdf1])
                                gdf.drop_duplicates(inplace=True)
                                gdf.to_file(self.city.gpkg, layer='gps_traces')
                    except error.HTTPError as e:
                        print("Download stopped; HTTP Error - %s" % e.code)
                        break
            return gdf

    def public_transit(self, run=True, date='2016-09-05'):
        if run:
            HOST = 'http://transit.land'

            def _request(uri):
                print(uri)
                req = request.Request(uri)
                req.add_header('Content-Type', 'application/json')
                response = request.urlopen(req)
                return json.loads(response.read())

            def request2(endpoint, **data):
                """Request with JSON response."""
                return _request(
                    '%s%s?%s' % (HOST, endpoint, parse.urlencode(data or {}))
                )

            def paginated(endpoint, key, **data):
                """Request with paginated JSON response. Returns generator."""
                response = request2(endpoint, **data)
                while response:
                    meta = response['meta']
                    print('%s: %s -> %s' % (
                        key,
                        meta['offset'],
                        meta['offset'] + meta['per_page']
                    ))
                    for entity in response[key]:
                        yield entity
                    if meta.get('next'):
                        response = _request(meta.get('next'))
                    else:
                        response = None

            def schedule_stop_pairs(**data):
                """Request Schedule Stop Pairs."""
                return paginated(
                    '/api/v1/schedule_stop_pairs',
                    'schedule_stop_pairs',
                    **data
                )

            def stops_f(**data):
                """Request Stops"""
                return paginated(
                    '/api/v1/stops',
                    'stops',
                    **data
                )

            def stop_f(onestop_id):
                """Request a Stop by Onestop ID."""
                return request2('/api/v1/stops/%s' % onestop_id)

            def duration(t1, t2):
                """Return the time between two HH:MM:SS times, in seconds."""
                fmt = '%H:%M:%S'
                t1 = datetime.datetime.strptime(t1, fmt)
                t2 = datetime.datetime.strptime(t2, fmt)
                return (t2 - t1).seconds

                ##########################################################
                ##### Count trips between stops, output GeoJSON      #####
                ##########################################################

            PER_PAGE = 500
            BBOX = list(self.city.boundary.bounds.transpose()[0])

            # [
            #     -118.266168,
            #     34.074479,
            #     -118.205915,
            #     34.037964
            # ]

            BETWEEN = [
                '01:00:00',
                '23:00:00'
            ]

            HOURS = duration(BETWEEN[0], BETWEEN[1]) / 3600.0
            # Minimum vehicles per hour
            # http://colorbrewer2.org/
            COLORMAP = {
                0: '#fef0d9',
                3: '#fdcc8a',
                6: '#fc8d59',
                10: '#d7301f'
            }

            OUTFILE = f"{self.city.directory}Databases/TransitLand/{self.city.municipality}.geojson"
            if os.path.exists(OUTFILE): os.remove(OUTFILE)

            # Group SSPs by (origin, destination) and count
            edges = {}
            ssps = schedule_stop_pairs(
                bbox=','.join(map(str, BBOX)),
                origin_departure_between=','.join(BETWEEN),
                date=date,
                per_page=PER_PAGE
            )
            for ssp in ssps:
                key = ssp['origin_onestop_id'], ssp['destination_onestop_id']
                if key not in edges:
                    edges[key] = 0
                edges[key] += 1

            # Get Stop geometries
            stops = {}
            for stop in stops_f(per_page=PER_PAGE, bbox=','.join(map(str, BBOX))):
                stops[stop['onestop_id']] = stop

            # Create GeoJSON Features
            colorkeys = sorted(COLORMAP.keys())
            features = []
            edges_sorted = sorted(list(edges.items()), key=lambda x: x[1])
            for (origin_onestop_id, destination_onestop_id), trips in edges_sorted:
                # Origin and destination geometries
                origin = stops.get(origin_onestop_id)
                destination = stops.get(destination_onestop_id)
                if not (origin and destination):
                    # Outside bounding box
                    continue
                # Frequency is in trips per hour
                frequency = trips / HOURS
                frequency_class = [i for i in colorkeys if frequency >= i][-1]
                print("Origin: %s Destination: %s Trips: %s Frequency: %s Freq. class: %s" % (
                    origin_onestop_id,
                    destination_onestop_id,
                    trips,
                    frequency,
                    frequency_class
                ))
                # Create the GeoJSON Feature
                features.append({
                    "type": "Feature",
                    "name": "%s -> %s" % (origin['name'], destination['name']),
                    "properties": {
                        "origin_onestop_id": origin_onestop_id,
                        "destination_onestop_id": destination_onestop_id,
                        "trips": trips,
                        "frequency": frequency,
                        "frequency_class": frequency_class,
                        "stroke": COLORMAP[frequency_class],
                        "stroke-width": frequency_class + 1,
                        "stroke-opacity": 1.0
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            origin['geometry']['coordinates'],
                            destination['geometry']['coordinates']
                        ]
                    }
                })

            # Create the GeoJSON Feature Collection
            fc = {
                "type": "FeatureCollection",
                "features": features
            }

            with open(OUTFILE, 'w', encoding='utf8') as outfile:
                json.dump(fc, outfile, sort_keys=True, indent=4, ensure_ascii=False)

            gdf = gpd.read_file(OUTFILE)

            try:
                gdf0 = gpd.read_file(self.city.gpkg, layer='public_transit')
                gdf = pd.concat([gdf0, gdf])
            except: pass

            gdf.to_file(self.city.gpkg, layer='public_transit')
            return print('> Public transit frequency downloaded and stored')

    def social_twitter(self):
        return self

    def social_instagram(self):
        return self

    # Scrape bylaws and legislations
    def bylaws(self, url):
        # Download files
        s = Scraper()
        s.get_all_files(url, prefix='Bylaws/'+self.city_name)

        # Classify files as bylaw, plan or guideline
        return print('')

    # Urban Codes
    def pdf_to_txt(self, path):
        # PDF to text using pdf miner
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        pagenos = set()

        with open(path, 'rb') as fp:
            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=False,
                                          check_extractable=False):
                interpreter.process_page(page)

        text = retstr.getvalue()

        fp.close()
        device.close()
        retstr.close()
        return text

    def bylaws_prescriptions(self):
        print('Bylaw prescriptions for ' + self.city_name)

        all_par = []
        all_class = []
        os.chdir("Bylaws/")

        for filename in glob.glob("*.pdf"):
            if self.city_name in filename:
                print(filename)
                # Tokenization into paragraphs
                raw_text = self.pdf_to_txt(filename)
                print(raw_text)
                text = raw_text.split('. \n')
                cl_text = []
                for t in text:
                    print(t)
                    cl_text.append(regex.sub(r"\p.{P}+", "", t))
                paragraphs = []
                for p in cl_text:
                    if len(p) > 1:
                        paragraphs.append(p)
                cl_paragraphs = []
                for n, p in zip(enumerate(paragraphs), paragraphs):
                    cl_paragraphs.append(str(n[0])+'__'+p.replace('\n', '')+'\n')
                    all_par.append(paragraphs)

                # Export tokens with id number
                try:
                    file = open(filename+'_p.txt', 'w')
                    file.writelines(cl_paragraphs)
                except:
                    file = open(filename + '_p.txt', 'w', encoding='utf-8')
                    file.writelines(cl_paragraphs)
                file.close()

                # Export classification files template in .txt for labelling if it does not exist
                filepath = filename+'_c.txt'
                if os.path.exists(filepath):
                    template = open(filepath, 'r+')
                else:
                    template = open(filepath, 'w')
                    template.close()
                    template = open(filepath, 'r+')
                lines = []
                if len(template.read().splitlines()) > 0:
                    print('Classification template not empty')
                else:
                    for n, p in zip(enumerate(cl_paragraphs), cl_paragraphs):
                        lines.append(str(n[0])+'__unclassified \n')
                    template.writelines(lines)
                template.close()

                # Using a text editor, manually label paragraphs into broader themes (density, use, parking, etc.)

                # Read manually labelled file
                classification = []
                c_file = open(filename+'_c.txt', 'r').read().splitlines()
                for line in c_file:
                    classification.append(line.split('__')[1])

                # Naive Bayes model
                def custom_tokenizer(str2):
                    lemmatizer = WordNetLemmatizer()
                    tokens = word_tokenize(str2)
                    remove_stopwords = list(filter(lambda token2: token2 not in stopwords.words("english"), tokens))
                    lematize_words = [lemmatizer.lemmatize(word2) for word2 in remove_stopwords]
                    return lematize_words

                vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
                tfidf = vectorizer.fit_transform(cl_paragraphs)
                print(tfidf)

                # Label Encoding
                le = LabelEncoder()
                le.fit(classification)
                all_class.append(le.classes_)
                print(le.classes_)
                # convert the classes into numeric value
                class_in_int = le.transform(classification)
                print(class_in_int)

                # Build the model

                # # Splitting into training and test dataset
                # x_train, x_test, y_train, y_test = train_test_split(tfidf, class_in_int, test_size=0.2, random_state=0)
                # classifier.fit(x_train, y_train)

                # Not splitting into training and test dataset
                classifier = MultinomialNB()
                classifier.fit(tfidf, class_in_int)

        # Break paragraph into sentences
        for cl in all_class:
            print(cl)
        for par in all_par:
            print(par)

        # Filter sentences according to theme

        # Label sentences with type of indicators (max_far, restr_uses, stalls_per_unit)

        # Build a supervised (Naive Bayes) model to predict themes and numeric data based on paragraph inputs

        """
        pred = classifier.predict(tfidf)
        print(metrics.confusion_matrix(class_in_int, pred), "\n")
        print(metrics.accuracy_score(class_in_int, pred))

        ## Other tutorials
        # Tokenization (grams)
        quotes_tokens = nltk.word_tokenize(raw_text)
        quotes_bigrams = list(nltk.bigrams(quotes_tokens))
        quotes_trigrams = list(nltk.trigrams(quotes_tokens))

        # Punctuation
        punctuation = re.compile(r'[-?!,:;()|]')
        post_punctuation = []
        for words in quotes_tokens:
            word = punctuation.sub("", words)
            if len(word)>0:
                post_punctuation.append(word)

        # Grammar tagging
        tagged = []
        for token in post_punctuation:
            tagged.append(nltk.pos_tag([token]))

        # Named entity recognition, NER
        ne_tokens = word_tokenize(raw_text)
        ne_tags = nltk.pos_tag(ne_tokens)
        ne_ner = ne_chunk(ne_tags)

        # Syntax trees
        new = "The big cat ate the little mouse who was after fresh cheese"
        new_tokens = nltk.pos_tag(word_tokenize(new))
        grammar_np = r"NP: {<DT>?<JJ>*<NN>}"
        chunk_parser = nltk.RegexpParser(grammar_np)
        chunk_result = chunk_parser.parse(new_tokens)

        # # Find paragraphs that contains numbers
        # n_paragraphs = []
        # for p in cl_paragraphs:
        #     if "m2" in p:
        #         n_paragraphs.append(p)

        # Cluster paragraphs into groups

        # (use, density, height, area, width, footprint, setbacks, yard, parking, bedroom)

        # def sent_tokenize(text):
        #     sentences = regex.split(r"[.!?]", text)
        #     sentences = [sent.strip(" ") for sent in sentences]
        #     return sentences
        #
        # print(sent_tokenize(text))

        # clean_text = regex.sub(r"\p{P}+", "", text)
        #
        # nlp = spacy.load('en_core_web_sm')
        # # doc = nlp(text)
        # # tokens = [token.lemma_ for token in doc]
        #
        # sentence_l = []
        # for sentence in sentences:
        #     sent = nlp(sentence)
        #     tokens = [token.lemma_ for token in sent]
        #     sentence_l.append(tokens)
        # sentence_l.append('floor area ratio')
        # vec = TfidfVectorizer()
        # features = vec.fit_transform(sentence_l)
        #
        # knn = NearestNeighbors(n_neighbors=10, metric='cosine')
        # knn.fit(features)
        # print(knn.kneighbors(features[0:1], return_distance=False))
        """
        return print('done')

    def bylaws_topics(self):
        from collections import Counter
        path = 'Bylaws/Burnaby_R1+Residential+District.pdf'

        # Sklearn
        from sklearn.feature_extraction.text import CountVectorizer
        # Plotting tools

        df = pd.read_csv('googleplaystore_user_reviews.csv', error_bad_lines = False)
        df = df.dropna(subset=['Translated_Review'])

        keywords = ['not']
        vectorizer = CountVectorizer(vocabulary=keywords, encoding='ISO-8859-1', analyzer='word',
                                     decode_error='ignore', ngram_range=(1, 1))
        dq = pd.DataFrame(columns=keywords)
        file = open(path, 'r', encoding="ISO-8859-1")
        print(file)

        """
        def custom_tokenizer(str2):
            lemmatizer = WordNetLemmatizer()
            tokens = nltk.word_tokenize(str2)
            remove_stopwords = list(filter(lambda token2: token2 not in stopwords.words("english"), tokens))
            lematize_words = [lemmatizer.lemmatize(word2) for word2 in remove_stopwords]
            return lematize_words

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
        """

        X = vectorizer.fit_transform(file)
        vocab = vectorizer.get_feature_names()
        counts = X.sum(axis=0).A1
        freq_distribution = Counter(dict(zip(vocab, counts)))
        allwords = dict(freq_distribution)
        totalnum = sum(allwords.values())
        allwords.update({'totalnum': totalnum})

        dy = pd.DataFrame.from_dict(allwords, orient='index')
        dy = dy.transpose()
        dq = dy.append(dq, sort=False)

        print(dq)
        return print(self.city_name + 'flexibility classified')

    # Data Visualization
    def display_results(self, files):
        print('')
        print('Displaying results')

        # Get results
        d = {}
        for i in files:
            try: d["{0}".format(i)] = gpd.read_file(i)
            except: d["{0}".format(i)] = i
            d["{0}".format(i)]['experiment'] = i
        all_exp = pd.concat(d)
        red = '#e2002b'
        yellow = '#ffb700'
        green = '#006500'
        blue = '#6a6ff1'
        purple = '#b959ff'
        columns = ['den_pop', 'den_ret', 'div_use', 'div_dwe', 'd2bike', 'd2bus', 'd2comm', 'd2OS', 'd2CV',
                   'den_route', 'inters_count']

        # Normalize columns
        all_exp = normalize(all_exp, columns)
        files_n = []
        for i in all_exp.experiment.unique():
            filepath = str(i) + '_n.geojson'
            files_n.append(filepath)
            if os.path.exists(filepath): print('Normalized file already exists!')
            else: all_exp[all_exp.experiment == i].to_file(filepath, driver='GeoJSON')
        d = {}
        for i in all_exp.experiment.unique():
            d["{0}".format(i)] = []
            for col in columns:
                d["{0}".format(i)].append(all_exp[all_exp.experiment == i][col].mean())
        n_cols = []
        for i in columns:
            n_cols.append('n_' + i)
        files_n.sort()

        # Generate livability wheels
        pc.generate_wheels(files_n, n_cols, filename='temp.html')
        return print('')


class Canada:
    def __init__(self, provinces):
        self.provinces=provinces
        self.cities = [province.cities for province in provinces]

    def update_databases(self, census=True):
        # Download dissemination areas from StatsCan
        if census:
            for province in self.provinces:
                for city in province.cities:
                    print(f"Downloading {city.city_name}'s dissemination area")
                    profile_url = "https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/hlt-fst/pd-pl/Tables/CompFile.cfm?Lang=Eng&T=1901&OFT=FULLCSV"
                    boundary_url = "http://www12.statcan.gc.ca/census-recensement/2011/geo/bound-limit/files-fichiers/2016/lda_000b16a_e.zip"

                    c_dir = f"{city.directory}StatsCan/"
                    if not os.path.exists(c_dir): os.makedirs(c_dir)

                    os.chdir(c_dir)

                    # Get data from StatCan webpage
                    download_file(profile_url, 'lda_profile.csv')
                    download_file(boundary_url)

                    # Open and reproject boundary file
                    bfilename = boundary_url.split('/')[-1][:-4]
                    archive = zipfile.ZipFile(f'{c_dir}{bfilename}.zip', 'r')
                    archive.extractall(c_dir)
                    census_da = gpd.read_file(f"{c_dir}{bfilename}.shp")
                    census_da.to_crs({'init': 'epsg:26910'}, inplace=True)

                    # Join DataFrames
                    df = pd.read_csv(f'{c_dir}lda_profile.csv', encoding="ISO-8859-1")
                    df['DAUID'] = df['Geographic code']
                    jda = census_da.merge(df, on='DAUID')

                    # Crop data to City boundary
                    city.DAs = gpd.sjoin(jda, city.boundary)

                    # Get Journey to Work data
                    if not os.path.exists(f"{c_dir}Mobility"): os.makedirs(f"{c_dir}Mobility")
                    os.chdir(f"{c_dir}Mobility")
                    mob_df = pd.DataFrame()
                    for da, csd in zip(city.DAs['DAUID'], city.DAs['CSDUID']):
                        print(f"Downloading Journey to Work data for DA: {csd}-{da}")
                        base_link = f'https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/prof/details/download-telecharger/current-actuelle.cfm?Lang=E&Geo1=DA&Code1={da}&Geo2=CSD&Code2={csd}&B1=Journey%20to%20work&type=0&FILETYPE=CSV'
                        try:
                            download_file(base_link, f"{csd}-{da}.csv")
                        except:
                            pass

                        # Preprocess Journey to Work data
                        df = pd.read_csv(f"{csd}-{da}.csv")
                        df = df.loc['Main mode of commuting']['Unnamed: 0']
                        dic = {i[0]: [i[2]] for i in df.index}
                        df = pd.DataFrame.from_dict(dic)
                        df['DAUID'] = da

                        # Append data to gdf
                        mob_df = pd.concat([mob_df, df])

                    # Join data to dissemination areas
                    city.DAs = city.DAs.merge(mob_df, on='DAUID')

                    # Save it to GeoPackage
                    city.DAs.to_file(city.gpkg, layer='land_dissemination_area')
                    print(f'Census dissemination area downloaded and saved at {city.gpkg}')


class BritishColumbia:
    def __init__(self, cities):
        self.cities = [GeoBoundary(f"{city}, British Columbia") for city in cities]

    def update_databases(self, icbc=True):

        # Check if ICBC crash data exists and join it from ICBC database if not
        if icbc:
            for city in self.cities:
                try:
                    city.crashes = gpd.read_file(city.gpkg, layer='network_accidents')
                    print(city.city_name + ' ICBC data read from database')
                except:
                    source = 'https://public.tableau.com/profile/icbc#!/vizhome/LowerMainlandCrashes/LMDashboard'
                    print('Adding ICBC crash data to ' + city.city_name + ' database')
                    df = city.merge_csv(f"{city.directory}Databases/ICBC/")
                    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
                    gdf = gpd.GeoDataFrame(df, geometry=geometry)
                    gdf.crs = 4326
                    gdf.to_crs(epsg=city.crs, inplace=True)
                    city.boundary.to_crs(epsg=city.crs, inplace=True)
                    matches = gpd.sjoin(gdf, city.boundary, op='within')
                    matches.to_file(city.gpkg, layer='icbc_accidents', driver='GPKG')

    # BC Assessment
    def aggregate_bca_from_field(self, run=True, inventory_dir='', geodatabase_dir=''):
        if run:
            for city in self.cities:
                print('Geoprocessing BC Assessment data from JUROL number')
                inventory = inventory_dir
                df = pd.read_csv(inventory)

                # Load and process Roll Number field on both datasets
                gdf = gpd.read_file(geodatabase_dir, layer='ASSESSMENT_FABRIC')

                # Reproject coordinate system
                gdf.crs = {'init': 'epsg:3005'}
                gdf.to_crs(epsg=city.crs, inplace=True)
                city.boundary.to_crs(epsg=city.crs, inplace=True)

                # Create spatial index and perform join
                s_index = gdf.sindex
                gdf = gpd.sjoin(gdf, city.boundary, op='within')

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
                print(len(out_gdf))

                # Reclassify land uses for BC Assessment data
                uses = {
                    'residential': ['000 - Single Family Dwelling', '030 - Strata-Lot Residence (Condominium)',
                                    '032 - Residential Dwelling with Suite',
                                    '033 - Duplex, Non-Strata Side by Side or Front / Back',
                                    '034 - Duplex, Non-Strata Up / Down', '035 - Duplex, Strata Side by Side',
                                    '036 - Duplex, Strata Front / Back', '039 - Row Housing (Single Unit Ownership)',
                                    '037 - Manufactured Home (Within Manufactured Home Park)',
                                    '038 - Manufactured Home (Not In Manufactured Home Park)',
                                    '040 - Seasonal Dwelling',
                                    '041 - Duplex, Strata Up / Down', '047 - Triplex', '049 - Fourplex',
                                    '050 - Multi-Family (Apartment Block)',
                                    '052 - Multi-Family (Garden Apartment & Row Housing)', '053 - Multi-Family (Conversion)',
                                    '054 - Multi-Family (High-Rise)', '055 - Multi-Family (Minimal Commercial)',
                                    '056 - Multi-Family (Residential Hotel)', '057 - Stratified Rental Townhouse',
                                    '058 - Stratified Rental Apartment (Frame Construction)',
                                    '059 - Stratified Rental Apartment (Hi-Rise Construction)',
                                    '060 - 2 Acres Or More (Single Family Dwelling, Duplex)', '285 - Seniors Licensed Care',
                                    '062 - 2 Acres Or More (Seasonal Dwelling)',
                                    '063 - 2 Acres Or More (Manufactured Home)',
                                    '234 - Manufactured Home Park',
                                    '286 - Seniors Independent & Assisted Living'],
                    'vacant': ['001 - Vacant Residential Less Than 2 Acres', '051 - Multi-Family (Vacant)',
                               '061 - 2 Acres Or More (Vacant)', '201 - Vacant IC&I',
                               '421 - Vacant', '422 - IC&I Water Lot (Vacant)',
                               '601 - Civic, Institutional & Recreational (Vacant)'],
                    'parking': ['020 - Residential Outbuilding Only', '029 - Strata Lot (Parking Residential)',
                                '043 - Parking (Lot Only, Paved Or Gravel-Res)', '219 - Strata Lot (Parking Commercial)',
                                '260 - Parking (Lot Only, Paved Or Gravel-Com)', '262 - Parking Garage',
                                '490 - Parking Lot Only (Paved Or Gravel)'],
                    'other': ['002 - Property Subject To Section 19(8)', '070 - 2 Acres Or More (Outbuilding)', '190 - Other',
                              '200 - Store(S) And Service Commercial', '205 - Big Box', '216 - Commercial Strata-Lot',
                              '220 - Automobile Dealership', '222 - Service Station', '224 - Self-Serve Service Station',
                              '226 - Car Wash', '227 - Automobile Sales (Lot)', '228 - Automobile Paint Shop, Garages, Etc.',
                              '230 - Hotel', '232 - Motel & Auto Court', '233 - Individual Strata Lot (Hotel/Motel)',
                              '237 - Bed & Breakfast Operation 4 Or More Units',
                              '239 - Bed & Breakfast Operation Less Than 4 Units',
                              '240 - Greenhouses And Nurseries (Not Farm Class)', '257 - Fast Food Restaurants',
                              '258 - Drive-In Restaurant', '288 - Sign Or Billboard Only'],
                    'retail': ['202 - Store(S) And Living Quarters', '209 - Shopping Centre (Neighbourhood)',
                               '211 - Shopping Centre (Community)', '212 - Department Store - Stand Alone',
                               '213 - Shopping Centre (Regional)', '214 - Retail Strip', '215 - Food Market',
                               '225 - Convenience Store/Service Station'],
                    'entertainment': ['236 - Campground (Commercial)', '250 - Theatre Buildings',
                                      '254 - Neighbourhood Pub', '256 - Restaurant Only',
                                      '266 - Bowling Alley', '270 - Hall (Community, Lodge, Club, Etc.)',
                                      '280 - Marine Facilities (Marina)',
                                      '600 - Recreational & Cultural Buildings (Includes Curling',
                                      '610 - Parks & Playing Fields', '612 - Golf Courses (Includes Public & Private)',
                                      '654 - Recreational Clubs, Ski Hills',
                                      '660 - Land Classified Recreational Used For'],
                    'civic': ['210 - Bank', '620 - Government Buildings (Includes Courthouse, Post Office',
                              '625 - Garbage Dumps, Sanitary Fills, Sewer Lagoons, Etc.', '630 - Works Yards',
                              '634 - Government Research Centres (Includes Nurseries &',
                              '640 - Hospitals (Nursing Homes Refer To Commercial Section).',
                              '642 - Cemeteries (Includes Public Or Private).',
                              '650 - Schools & Universities, College Or Technical Schools',
                              '652 - Churches & Bible Schools'],
                    'agriculture': ['110 - Grain & Forage', '120 - Vegetable & Truck',
                                    '150 - Beef', '170 - Poultry', '180 - Mixed'],
                    'office': ['203 - Stores And/Or Offices With Apartments', '204 - Store(S) And Offices',
                               '208 - Office Building (Primary Use)'],
                    'industrial': ['217 - Air Space Title', '272 - Storage & Warehousing (Open)',
                                   '273 - Storage & Warehousing (Closed)', '274 - Storage & Warehousing (Cold)',
                                   '275 - Self Storage', '276 - Lumber Yard Or Building Supplies', '400 - Fruit & Vegetable',
                                   '401 - Industrial (Vacant)', '402 - Meat & Poultry', '403 - Sea Food',
                                   '404 - Dairy Products', '405 - Bakery & Biscuit Manufacturing',
                                   '406 - Confectionery Manufacturing & Sugar Processing', '408 - Brewery',
                                   '414 - Miscellaneous (Food Processing)',
                                   '416 - Planer Mills (When Separate From Sawmill)',
                                   '419 - Sash & Door',
                                   '420 - Lumber Remanufacturing (When Separate From Sawmill)',
                                   '423 - IC&I Water Lot (Improved)',
                                   '424 - Pulp & Paper Mills (Incl Fine Paper, Tissue & Asphalt Roof)',
                                   '425 - Paper Box, Paper Bag, And Other Paper Remanufacturing.', '428 - Improved',
                                   '429 - Miscellaneous (Forest And Allied Industry)',
                                   '434 - Petroleum Bulk Plants',
                                   '445 - Sand & Gravel (Vacant and Improved)',
                                   '447 - Asphalt Plants',
                                   '448 - Concrete Mixing Plants',
                                   '449 - Miscellaneous (Mining And Allied Industries)', '452 - Leather Industry',
                                   '454 - Textiles & Knitting Mills', '456 - Clothing Industry',
                                   '458 - Furniture & Fixtures Industry', '460 - Printing & Publishing Industry',
                                   '462 - Primary Metal Industries (Iron & Steel Mills,', '464 - Metal Fabricating Industries',
                                   '466 - Machinery Manufacturing (Excluding Electrical)',
                                   '470 - Electrical & Electronics Products Industry',
                                   '472 - Chemical & Chemical Products Industries', '474 - Miscellaneous & (Industrial Other)',
                                   '476 - Grain Elevators', '478 - Docks & Wharves', '500 - Railway',
                                   '505 - Marine & Navigational Facilities (Includes Ferry',
                                   '510 - Bus Company, Including Street Railway', '520 - Telephone',
                                   '530 - Telecommunications (Other Than Telephone)',
                                   '550 - Gas Distribution Systems',
                                   '560 - Water Distribution Systems',
                                   '580 - Electrical Power Systems (Including Non-Utility']
                }
                new_uses = []
                index = list(out_gdf.columns).index("PRIMARY_ACTUAL_USE")
                all_prim_uses = [item for sublist in list(uses.values()) for item in sublist]
                for row in out_gdf.iterrows():
                    for key, value in uses.items():
                        if row[1]['PRIMARY_ACTUAL_USE'] in value:
                            new_uses.append(key)
                    if row[1]['PRIMARY_ACTUAL_USE'] not in all_prim_uses:
                        new_uses.append(row[1]['PRIMARY_ACTUAL_USE'])
                out_gdf['elab_use'] = new_uses

                # Export assessment fabric layer to GeoPackage
                out_gdf.to_file(city.gpkg, driver='GPKG', layer='land_assessment_fabric')

                # Delete repeated parcels
                p_gdf = out_gdf.drop_duplicates(subset=['geometry'])

                # Classify parcels into categories
                p_gdf['area'] = p_gdf.geometry.area
                p_gdf.loc[p_gdf['area'] < 400, 'elab_size'] = '-400'
                p_gdf.loc[(p_gdf['area'] > 400) & (p_gdf['area'] < 800), 'elab_size'] = '400-800'
                p_gdf.loc[(p_gdf['area'] > 800) & (p_gdf['area'] < 1600), 'elab_size'] = '800-1600'
                p_gdf.loc[(p_gdf['area'] > 1600) & (p_gdf['area'] < 3200), 'elab_size'] = '1600-3200'
                p_gdf.loc[(p_gdf['area'] > 3200) & (p_gdf['area'] < 6400), 'elab_size'] = '3200-6400'
                p_gdf.loc[p_gdf['area'] > 6400, 'elab_size'] = '6400-'

                # Export parcel layer to GeoPackage
                p_gdf.to_file(city.gpkg, driver='GPKG', layer='land_assessment_parcels')
                return {'properties': out_gdf, 'parcels': p_gdf}


class Vancouver:
    """Web scraping Vancouver OpenData"""
    def __init__(self, directory):
        self.directory = directory
        return

    def get_permits(self):
        # Get list of pdfs from ongoing development permits at City of Vancouver webpage
        url = 'https://development.vancouver.ca/'
        s = Scraper()
        links = s.get_all_links(url)
        pdfs = {}
        for link in links:
            if link != None:
                if ('.pdf' in link) or ('.PDF' in link):
                    pass
                else:
                    try: name = link.split(url)[1]
                    except: name = 'unknown'
                    pdfs[name] = []
                    sublinks = s.get_all_links(link)
                    if sublinks is not None:
                        for sublink in sublinks:
                            if sublink is None:
                                pass
                            elif '.pdf' in sublink:
                                pdfs[name].append(sublink)
        s.close_session()

        # Download pdfs and move it to a folder
        for key, value in pdfs.items():
            for pdf in value:
                try: pieces = pdf.split(url)[1]
                except: pieces = pdf
                pieces = pieces.split('/')
                filename = '_'.join(pieces)
                response = requests.get(pdf)
                with open(f'{self.directory}{filename}', 'wb') as f:
                    f.write(response.content)
                print(f"{filename} saved")
        print(str(len(pdfs)) + ' permit documents downloaded at ' + str(datetime.datetime.now()))
        return
