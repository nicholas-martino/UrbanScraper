# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Nicholas Martino

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import datetime
import gc
import glob
import io
import json
import logging
import os
import random
import re
import time
import zipfile
from io import StringIO
from urllib import request, parse

import geopandas as gpd
import osmnx as ox
import pandas as pd
import regex
import requests
import selenium.webdriver as webdriver

from bs4 import BeautifulSoup
from craigslist import CraigslistHousing
from geopy.geocoders import Nominatim
from lxml import html
from nltk import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from pdfminer.converter import TextConverter
# from pdfminer.layout import LAParams
# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.pdfpage import PDFPage
from selenium.webdriver.firefox.options import Options
from shapely import wkt
from shapely.geometry import *
from skbio import diversity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
# from spacy.util import minibatch, compounding


def download_file(url, file_path):
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return file_path


class Scraper:
    def __init__(self, driver='/opt/anaconda3/bin/geckodriver', timeout=0):
        options = Options()
        options.set_preference("browser.download.folderList", 1)
        options.set_preference("browser.download.manager.showWhenStarting", False)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
        options.set_preference("pdfjs.disabled", True)
        driver = webdriver.Firefox(executable_path=driver, options=options)
        driver.minimize_window()
        if timeout != 0:
            driver.set_page_load_timeout(timeout)
        self.driver = driver

    def get_sublinks(self, url):
        # Get all link elements at a given url
        try:
            self.driver.get(url)
            self.driver.implicitly_wait(1)
            elements = self.driver.find_elements_by_tag_name('a')
            links = []
            for element in elements:
                try:
                    link = element.get_attribute('href')
                    links.append(link)
                except: pass
            for link in links:
                if (link == None) or (link == 'None'):
                    links.remove(link)
            return links
        except:
            return None

    def download_files(self, url, path, format='.pdf'):
        # Get all files of a given format at a given url
        links = self.get_sublinks(url)
        filenames = []
        for link in links:
            if format in link:
                directory = path.split('/')[0]
                local_filename = path + '_' + url.split('/')[-1]
                filenames.append(local_filename)
                with requests.get(link, stream=True) as r:
                    r.raise_for_status()
                    with open(local_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
        return filenames

    def close_session(self):
        self.driver.close()


class GeoScraper:
    """Web scraping GeoSpatial data"""
    def __init__(self, city, directory='/Volumes/Samsung_T5/Databases/', crs=26910):
        self.city = city
        self.dir = directory
        self.crs = crs
        print(f"Scraping data for elements within {city}\n")

    # Scrape built environment
    def buildings_osm(self):
        return self

    def blocks_osm(self):
        return

    # Scrape affordability and vitality indicators
    def employment_craigslist(self):
        return self

    def employment_indeed(self, run=True, n_pages=1):
        if run:
            print(f"> Downloading employment posts from Indeed")

            # Build search
            name = self.city.municipality.replace(',', '%2C').replace(' ', '+')
            start = 0

            # Create selenium driver
            s = Scraper()
            s.driver.minimize_window()

            data = {'job': [], 'time': [], 'salary': [], 'company': [], 'geometry': []}
            for i in range(start, n_pages):
                url = f"https://www.indeed.ca/jobs?q=&l={name}&start={i*10}"

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
                        footer = soup1.find_all(name="div", attrs={"class": "jobsearch-JobMetadataFooter"})
                        if len(footer) == 0: data['time'].append('Unknown')
                        else:
                            for div1 in footer:
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
                c_links = gpd.read_file(f"{self.dir}{self.city.municipality}.gpkg", layer='network_links', driver='GPKG')
                cl_strs = [i.replace("Avenue", "").replace("Street", "").replace("Road", "").replace("Drive", "").strip(" ")
                           for i in list(c_links) if i is not None]

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
                    while True:
                        try:
                            s.driver.get(url)
                            time.sleep(2)
                            break
                        except: pass

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
                        try: geom = Point(locator.geocode(f"{address}, {self.city.municipality}")[1])
                        except: geom = 'Unknown'
                    data['geometry'].append(geom)

            # Close web browser
            s.close_session()

            # Export to GeoPackage
            gdf = gpd.GeoDataFrame().from_dict(data)
            gdf = gdf.loc[gdf['geometry'] != 'Unknown']
            gdf['geometry'] = [Point(t.y, t.x) for t in gdf.geometry]
            gdf = gdf.set_geometry('geometry')
            gdf.crs = 4326
            gdf.to_crs(epsg=self.crs, inplace=True)
            # self.city.boundary.to_crs(epsg=self.crs, inplace=True)
            gdf['date'] = str(datetime.datetime.now().date())

            gdf0 = gpd.read_file(f"{self.dir}Indeed/{self.city.municipality}.geojson", driver='GeoJSON', encoding="ISO-8859-1").copy()
            gdf = pd.concat([gdf0, gdf]).reset_index(drop=True)

            def range_mean(rng):
                if ',' in rng: rng = rng.replace(',', '')
                numerical = [float(r.split('$')[1]) if '$' in r else float(r) for r in rng.split(' - ')]
                return sum(numerical) / len(numerical)

            # Convert income data to numerical
            for i in gdf.index:
                if gdf.at[i, 'salary'] != 'Unknown':
                    for time_frame, den in {' a year': 1, ' a month': 12, ' a week': 40, ' a day': 200, ' an hour': (6 * 200)}.items():
                        if time_frame in gdf.at[i, 'salary']:
                            salary = gdf.at[i, 'salary'].strip().split(time_frame)[0]
                            for j in [',', '\n']:
                                if j in salary: salary = salary.replace(j, '')
                            if ' - ' in salary: salary = range_mean(salary) * den
                            else: salary = float(salary.split("$")[1]) * den
                    gdf.at[i, 'salary_n'] = salary

            gdf = gdf.loc[:, ['job', 'time', 'salary', 'salary_n', 'company', 'date', 'geometry']]
            gdf = gdf.drop_duplicates(subset=['job'])
            gdf = gdf.reset_index(drop=True)
            if len(gdf) > 0:
                gdf.to_file(f"{self.dir}/Indeed/{self.city.municipality}.geojson", driver='GeoJSON')
                try:
                    gdf.to_file(f"{self.dir}{self.city.municipality}.gpkg", layer='indeed_employment', driver='GPKG')
                    print(f"> Employment data downloaded from Indeed and saved on {self.city.municipality} database with {len(gdf)} data points")
                except: print(f"!!! Employment data from Indeed not saved on GeoPackage !!!")
            return gdf

    def housing_craigslist(self, site, n_results, run=True):
        print(f"Downloading {self.city.municipality}'s housing posts from Craigslist")

        if run:
            try:
                cl = CraigslistHousing(site=site)
                results = cl.get_results(sort_by='newest', geotagged=True, limit=n_results)
            except:
                print("!!! Craigslist download failed :(")
                results = None

            if results is not None:
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
                    try: price.append(float(result[('price')][1:]))
                    except: price.append(float(result[('price')][1:].replace(',','')))
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

                # Filter area and bedrooms None
                cl_gdf = cl_gdf[(cl_gdf['area'] != None) & (cl_gdf['bedrooms'] != None)]

                # Calculate Price/Area
                cl_gdf['area'] = cl_gdf['area'].astype(float)
                cl_gdf['price_sqft'] = cl_gdf['price'] / cl_gdf['area']

                # Get Vancouver Boundary
                van_gdf = ox.geocode_to_gdf(self.city.municipality)
                van_gdf.crs = 4326
                van_gdf.to_crs(epsg=self.crs)
                van_pol = van_gdf.geometry[0]
                if van_pol is None: print('Administrative boundary download failed :(')

                # Filter data
                pd.set_option('display.min_rows', 10)
                cl_gdf.to_crs(epsg=self.crs)
                cl_gdf = cl_gdf[cl_gdf.within(van_pol)]
                csv_path = '__pycache__/' + self.city.municipality + '_Craigslist.csv'
                cl_gdf = cl_gdf[cl_gdf['area'] > 270]

                # Write data in csv
                with io.open(csv_path, "a", encoding="utf-8") as f:
                    cl_gdf.to_csv(f, header=False, index=False)
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

                # Read existing data
                try: cl_cur = gpd.read_file(f"{self.dir}{self.city.municipality}.gpkg", layer='craigslist_housing', driver='GPKG')
                except: cl_cur = gpd.GeoDataFrame()
                try: cl_cur.drop('fid', inplace=True)
                except: pass
                gdf = pd.concat([cl_cur, gdf])
                gdf.drop_duplicates(inplace=True)
                print(f"Craigslist data for {self.city.municipality} downloaded and joined with {len(gdf)} data points :)")

                # Separate buy and rent posts
                gdf.to_file(f"{self.dir}{self.city.municipality}.gpkg", layer='craigslist_housing', driver="GPKG")
                gdf = gdf[(gdf['price'] > 100) & (gdf['price'] < 50000)]
                gdf.to_file(f"{self.dir}{self.city.municipality}.gpkg", layer='craigslist_rent', driver="GPKG")
                print(f"Craigslist data for {self.city.municipality} exported to GeoPackage")

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
    def air_quality(self, token, run=True):
        """
        A Python wrapper for AQICN API.
        The library can be used to search and retrieve Air Quality Index data.
        Please refer to AQICN website to obtain token that must be used for access.
        """
        if run:
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
            try:
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
            except: print(f"Air quality data scraping failed for {self.city.municipality}")

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

            try: gdf = gpd.read_file(self.city.gpkg, layer='gps_traces', driver='GPKG', encoding='ISO-8859-1')
            except: gdf = gpd.GeoDataFrame(columns=['file'])

            files = []
            for i, b in enumerate(boundaries):
                min_ln, min_lt, max_ln, max_lt = b.bounds

                # Verify existing files
                try: init = int(max([float(k[2:-3]) for k in os.listdir(directory) if f"{i}_" in k])) + 1
                except: init = 0

                # Start downloading
                for j in range(init, 100000):
                    page = j
                    file_name = f"{i}_{page}.gpx"
                    url = f"http://api.openstreetmap.org/api/0.6/trackpoints?bbox={min_ln},{min_lt},{max_ln},{max_lt}&page={page}"

                    if file_name in gdf['file'].unique() or file_name in os.listdir(directory):
                        print(f"> {file_name} found, passing to next file")
                    else:
                        try:
                            # Scrape data from OSM
                            print(f"> Saving trace {file_name} from {url}")
                            u = request.urlopen(url)
                            buffer = u.read()

                            # Write requested data into file
                            if len(buffer) < 300:
                                print(f"File size of {len(buffer)} is too small, jumping to next space")
                                break
                            else:
                                f = open(f"{directory}/{file_name}", 'wb')
                                f.write(buffer)
                                f.close()
                                files.append(file_name)
                                time.sleep(1)
                                gc.collect()
                        except: print("> Traces downloaded, but not written to database")

            len_un = len(gdf['file'].unique())
            print(f"\n{len_un} traces found in the database")

            files_list = os.listdir(directory)
            for i, file in enumerate(files_list):
                if file not in gdf['file'].unique():
                    # Read and reproject downloaded trace
                    try:
                        print(f"> Reading and reprojecting trace from file {file} ({i+1}/{len(files_list)-len_un})")
                        gpx = gpd.read_file(f"{directory}/{file}", layer='track_points', encoding='ISO-8859-1')
                        gpx['file'] = file
                        gpx.crs = 4326
                        gpx = gpx.to_crs(epsg=self.city.crs)
                        gdf = pd.concat([gdf, gpx])
                    except: print(f"> Failed to read or reproject traces from file {file}")

                    # Merge into existing database
                    gdf = gdf.reset_index(drop=True)
                    gc.collect()

            try:
                gdf = gdf.drop_duplicates()
                gdf = gdf[gdf.geometry.is_valid]
                print(f"> Writing downloaded data to GeoPackage")
                gdf.to_file(self.city.gpkg, layer='gps_traces', driver='GPKG', encoding='ISO-8859-1')
            except: print(f"!!! Failed to write data to GeoPackage !!!")
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
                gdf0 = gpd.read_file(self.city.gpkg, layer='public_transit', driver='GPKG')
                gdf = pd.concat([gdf0, gdf])
            except: pass

            gdf.to_file(self.city.gpkg, layer='public_transit', driver='GPKG')
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
                    print(f"Downloading {city.city_name}'s dissemination areas from StatsCan")
                    profile_url = "https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/hlt-fst/pd-pl/Tables/CompFile.cfm?Lang=Eng&T=1901&OFT=FULLCSV"
                    boundary_url = "http://www12.statcan.gc.ca/census-recensement/2011/geo/bound-limit/files-fichiers/2016/lda_000b16a_e.zip"

                    c_dir = f"{city.directory}/Databases/StatsCan/"
                    if not os.path.exists(c_dir): os.makedirs(c_dir)

                    os.chdir(c_dir)

                    # Get data from StatCan webpage
                    download_file(profile_url, 'lda_profile.csv')
                    download_file(boundary_url, 'lda_000b16a_e.zip')

                    # Open and reproject boundary file
                    bfilename = boundary_url.split('/')[-1][:-4]
                    archive = zipfile.ZipFile(f'{c_dir}{bfilename}.zip', 'r')
                    archive.extractall(c_dir)
                    census_da = gpd.read_file(f"{c_dir}{bfilename}.shp")
                    census_da.to_crs(city.crs, inplace=True)

                    # Join DataFrames
                    df = pd.read_csv(f'{c_dir}lda_profile.csv', encoding="ISO-8859-1")
                    df['DAUID'] = df['Geographic code']
                    df = df.rename(columns={"Population density per square kilometer, 2016": "population_den",
                        "Total private dwellings, 2016": "n_dwellings"})
                    jda = census_da.merge(df, on='DAUID')

                    # Crop data to City boundary
                    city.boundary.to_crs(city.crs, inplace=True)
                    city.DAs = gpd.sjoin(jda, city.boundary)

                    # Get Journey to Work data
                    if not os.path.exists(f"{c_dir}DAs"): os.makedirs(f"{c_dir}DAs")
                    os.chdir(f"{c_dir}DAs")
                    out_df = pd.DataFrame()
                    for i, (da, csd) in enumerate(zip(city.DAs['DAUID'], city.DAs['CSDUID'])):
                        print(f"DA: {da} ({i+1} / {len(city.DAs['DAUID'])+1})")
                        for d in ['Education', 'Ethnic%20Origin', 'Housing', 'Income', 'Journey%20to%20work', 'Labour']:
                            print(f"> Downloading {d}")
                            base_url = f'https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/prof/details/download-telecharger/current-actuelle.cfm?Lang=E&Geo1=DA&Code1={da}&Geo2=CSD&Code2={csd}&B1={d}&type=0&FILETYPE=CSV'
                            try: download_file(base_url, f"{csd}-{da}-{d}.csv")
                            except:
                                print(f"> Download from {d} data failed for DA {da} at CSD {csd}")
                                pass

                        # Pre process base funcion
                        clean = lambda df: df['Unnamed: 1'].reset_index().loc[:, ['level_1', 'level_3']]\
                            .iloc[1:].set_index('level_1').transpose().reset_index(drop=True)

                        # Pre process Education
                        df = pd.read_csv(f"{csd}-{da}-Education.csv", encoding='ISO-8859-1')
                        df_ed = clean(df)
                        df_ed_degrees = df_ed.iloc[:, 1:5]
                        df_ed_areas = df_ed.iloc[: , 32:93].loc[:, [
                            '  No postsecondary certificate; diploma or degree',
                            '  Education',
                            '  Visual and performing arts; and communications technologies',
                            '  Humanities', '  Social and behavioural sciences and law',
                            '  Business; management and public administration',
                            '  Physical and life sciences and technologies',
                            '  Mathematics; computer and information sciences',
                            '  Architecture; engineering; and related technologies',
                            '  Agriculture; natural resources and conservation',
                            '  Health and related fields',
                            '  Personal; protective and transportation services',
                            '  Other'
                        ]]
                        df_ed = pd.concat([df_ed_degrees, df_ed_areas], axis=1)
                        df_ed = df_ed.rename(columns={
                            'Total - Highest certificate; diploma or degree for the population aged 15 years and over in private households - 25% sample data': 'ed_total',
                            '  No certificate; diploma or degree': 'no_certificate',
                            '  Secondary (high) school diploma or equivalency certificate': 'secondary',
                            '  Postsecondary certificate; diploma or degree': 'postsecondary',
                            '  No postsecondary certificate; diploma or degree': 'no_postsecondary',
                            '  Education': 'education',
                            '  Visual and performing arts; and communications technologies': 'arts',
                            '  Humanities': 'humanities',
                            '  Social and behavioural sciences and law': 'social',
                            '  Business; management and public administration': 'business',
                            '  Physical and life sciences and technologies': 'natural',
                            '  Mathematics; computer and information sciences': 'information',
                            '  Architecture; engineering; and related technologies': 'engineering',
                            '  Agriculture; natural resources and conservation': 'agriculture',
                            '  Health and related fields': 'health',
                            '  Personal; protective and transportation services': 'protective',
                            '  Other': 'other'
                        }).astype(float, errors='ignore')
                        df_ed['no_certificate_ratio'] = df_ed['no_certificate']/df_ed['ed_total']
                        df_ed['secondary_ratio'] = df_ed['secondary']/df_ed['ed_total']
                        df_ed['postsecondary_ratio'] = df_ed['postsecondary']/df_ed['ed_total']
                        df_ed['no_postsecondary_ratio'] = df_ed['no_postsecondary']/df_ed['ed_total']
                        df_ed_div = df_ed.loc[:, ['education', 'arts', 'humanities', 'social', 'business', 'natural',
                            'information', 'engineering', 'agriculture', 'health', 'protective']]
                        df_ed['educ_div_sh'] = diversity.alpha_diversity('shannon', df_ed_div)
                        df_ed['educ_div_si'] = diversity.alpha_diversity('simpson', df_ed_div)
                        df_ed['DAUID'] = da

                        # Pre process Ethnic Origins
                        profile = download_file(f'https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/prof/details/download-telecharger/current-actuelle.cfm?Lang=E&Geo1=DA&Code1={da}&Geo2=CSD&Code2={csd}&B1=All&type=0&FILETYPE=CSV', f"{csd}-{da}-Profile.csv")
                        df = pd.read_csv(profile, encoding='ISO-8859-1', error_bad_lines=False)
                        df_age1 = df.loc['Age characteristics'].iloc[2:4]
                        df_age2 = df.loc['Age characteristics'].iloc[6:15]
                        df_age3 = df.loc['Age characteristics'].iloc[17:21]
                        df_age4 = df.loc['Age characteristics'].iloc[23:26]
                        df_age = pd.concat([df_age1, df_age2, df_age3, df_age4])
                        counts = []
                        for i in df_age.index:
                            try: counts.append(int(i[2]))
                            except: counts.append(0)
                        df_ag = pd.DataFrame()
                        if sum(counts) == 0:
                            df_ag.at[0, 'age_div_sh'] = 0
                            df_ag.at[0, 'age_div_sh'] = 0
                        else:
                            df_ag.at[0, 'age_div_sh'] = diversity.alpha_diversity('shannon', counts)[0]
                            df_ag.at[0, 'age_div_sh'] = diversity.alpha_diversity('simpson', counts)[0]
                        df_ag['DAUID'] = da

                        indices = [1, 5, 15, 89, 110, 135, 203, 269]
                        df_eth = df.loc['Ethnic origin population'].iloc[indices]
                        df_et = pd.DataFrame()
                        counts = []
                        for i in df_eth.index:
                            try: counts.append(int(i[2]))
                            except: counts.append(0)
                        if sum(counts) == 0:
                            df_et.at[0, 'ethnic_div_sh'] = 0
                            df_et.at[0, 'ethnic_div_si'] = 0
                        else:
                            df_et.at[0, 'ethnic_div_sh'] = diversity.alpha_diversity('shannon', counts)[0]
                            df_et.at[0, 'ethnic_div_si'] = diversity.alpha_diversity('simpson', counts)[0]
                        df_et['DAUID'] = da

                        # Pre process Housing
                        df = pd.read_csv(f"{csd}-{da}-Housing.csv", encoding='ISO-8859-1')
                        df_ho = clean(df)
                        df_ho_gen = df_ho.iloc[:, 1:35]
                        df_ho_aff = df_ho.iloc[:, 58:67]
                        df_ho = pd.concat([df_ho_gen, df_ho_aff], axis=1)
                        # Filter null columns
                        df_ho = df_ho.loc[:, df_ho.columns.notnull()]
                        # Clean columns names
                        n_cols = []
                        for col in df_ho.columns:
                            try: n_cols.append(col.strip())
                            except: n_cols.append(col)
                        df_ho.columns = n_cols
                        # Rename columns and convert to numeric
                        df_ho = df_ho.rename(columns={
                            'Total - Private households by tenure - 25% sample data':'total_tenure',
                            'Total - Occupied private dwellings by condominium status - 25% sample data':'total_cond',
                            'Total - Occupied private dwellings by number of bedrooms - 25% sample data':'total_bedr',
                            'Total - Occupied private dwellings by number of rooms - 25% sample data':'total_rooms',
                            'Average number of rooms per dwelling':'ave_n_rooms',
                            'Total - Private households by number of persons per room - 25% sample data':'total_people_per_room',
                            'Total - Private households by housing suitability - 25% sample data':'total_suitability',
                            'Total - Occupied private dwellings by period of construction - 25% sample data':'total_period',
                            'Median monthly shelter costs for owned dwellings ($)':'owned_med_cost',
                            'Average monthly shelter costs for owned dwellings ($)':'owned_ave_cost',
                            'Median value of dwellings ($)':'owned_med_dwe_value',
                            'Average value of dwellings ($)':'owned_ave_dwe_value',
                            'Total - Tenant households in non-farm; non-reserve private dwellings - 25% sample data':'total_tenant',
                            '% of tenant households in subsidized housing':'receives_subsidy_rat',
                            '% of tenant households spending 30% or more of its income on shelter costs':'more30%income_rat',
                            'Median monthly shelter costs for rented dwellings ($)':'rented_med_cost',
                            'Average monthly shelter costs for rented dwellings ($)':'rented_ave_cost'
                            }).astype(float, errors='ignore')
                        # Calculate ratios
                        df_ho['owner_ratio'] = df_ho['Owner']/df_ho['total_tenure']
                        df_ho['renter_ratio'] = df_ho['Renter']/df_ho['total_tenure']
                        df_ho['condominium_ratio'] = df_ho['Condominium']/df_ho['total_cond']
                        df_ho['not_condominium_ratio'] = df_ho['Not condominium']/df_ho['total_cond']
                        df_ho['no_bedrooms_ratio'] = df_ho['No bedrooms']/df_ho['total_bedr']
                        df_ho['1_bedroom_ratio'] = df_ho['1 bedroom']/df_ho['total_bedr']
                        df_ho['2_bedrooms_ratio'] = df_ho['2 bedrooms']/df_ho['total_bedr']
                        df_ho['3_bedrooms_ratio'] = df_ho['3 bedrooms']/df_ho['total_bedr']
                        df_ho['4_plus_bedrooms_ratio'] = df_ho['4 or more bedrooms']/df_ho['total_bedr']
                        df_ho['1_4_rooms_ratio'] = df_ho['1 to 4 rooms']/df_ho['total_rooms']
                        df_ho['5_rooms_ratio'] = df_ho['5 rooms']/df_ho['total_rooms']
                        df_ho['6_rooms_ratio'] = df_ho['6 rooms']/df_ho['total_rooms']
                        df_ho['7_rooms_ratio'] = df_ho['7 rooms']/df_ho['total_rooms']
                        df_ho['8_plus_rooms_ratio'] = df_ho['8 or more rooms']/df_ho['total_rooms']
                        df_ho['1_person_per_room_ratio'] = df_ho['One person or fewer per room']/df_ho['total_people_per_room']
                        df_ho['1_plus_person_per_room_ratio'] = df_ho['More than 1 person per room']/df_ho['total_people_per_room']
                        df_ho['suitable_ratio'] = df_ho['Suitable']/df_ho['total_suitability']
                        df_ho['not_suitable_ratio'] = df_ho['Not suitable'] / df_ho['total_suitability']
                        df_bad = df_ho.loc[:, ['1960 or before', '1961 to 1980', '1981 to 1990', '2001 to 2005',
                            '2006 to 2010', '2011 to 2016']]
                        df_ho['building_age_div_si'] = diversity.alpha_diversity("simpson", df_bad)
                        df_ho['building_age_div_sh'] = diversity.alpha_diversity("shannon", df_bad)
                        df_ddr = df_ho.loc[:, ['1 to 4 rooms', '5 rooms', '6 rooms', '7 rooms', '8 or more rooms']]
                        df_ho['dwelling_div_rooms_si'] = diversity.alpha_diversity("simpson", df_ddr)
                        df_ho['dwelling_div_rooms_sh'] = diversity.alpha_diversity("shannon", df_ddr)
                        df_ddb = df_ho.loc[:, ['No bedrooms', '1 bedroom', '2 bedrooms',
                            '3 bedrooms', '4 or more bedrooms']]
                        df_ho['dwelling_div_bedrooms_si'] = diversity.alpha_diversity("simpson", df_ddb)
                        df_ho['dwelling_div_bedrooms_sh'] = diversity.alpha_diversity("shannon", df_ddb)
                        df_ho['DAUID'] = da

                        # Pre process Income
                        df = pd.read_csv(f"{csd}-{da}-Income.csv", encoding='ISO-8859-1')
                        df_in = clean(df)
                        df_in_stats = df_in.iloc[:, 2:13]
                        df_in_empl_stats = df_in.iloc[:, 28:35]
                        df_in_grp_aft_tax = df_in.iloc[:, 48:64]
                        df_in = pd.concat([df_in_stats, df_in_empl_stats, df_in_grp_aft_tax], axis=1)
                        df_in = df_in.rename(columns={
                            '  Number of total income recipients aged 15 years and over in private households - 100% data':'income_recipients',
                            '    Median total income in 2015 among recipients ($)':'median_income',
                            '  Number of after-tax income recipients aged 15 years and over in private households - 100% data':'after_tax_recipients',
                            '    Median after-tax income in 2015 among recipients ($)':'med_income_after_tax',
                            '  Number of market income recipients aged 15 years and over in private households - 100% data':'mkt_recipients',
                            '    Median market income in 2015 among recipients ($)':'med_mkt_income',
                            '  Number of government transfers recipients aged 15 years and over in private households - 100% data':'n_gov_transfers',
                            '    Median government transfers in 2015 among recipients ($)':'median_gov_transfers',
                            '  Number of employment income recipients aged 15 years and over in private households - 100% data':'n_emp_income',
                            '    Median employment income in 2015 among recipients ($)':'med_emp_income',
                            'Total - Income statistics in 2015 for the population aged 15 years and over in private households - 25% sample data':'total_income_stats',
                            '  Market income (%)':'mkt_income',
                            '    Employment income (%)':'emp_income',
                            '  Government transfers (%)':'gov_transfers',
                            'Total - Total income groups in 2015 for the population aged 15 years and over in private households - 100% data':'total_income_groups',
                            '  Without total income':'without_income',
                            '  With total income':'with_income',
                            '  Percentage with total income':'income_ratio',
                            'Total - After-tax income groups in 2015 for the population aged 15 years and over in private households - 100% data':'total_income_after_tax',
                            '  Without after-tax income':'without_income_at',
                            '  With after-tax income':'with_income_at',
                            '  Percentage with after-tax income':'income_ratio_at'
                        }).astype(float, errors='ignore')
                        df_dic = df_in.loc[:, [
                            '    Under $10;000 (including loss)',
                            '    $10;000 to $19;999',
                            '    $20;000 to $29;999',
                            '    $30;000 to $39;999',
                            '    $40;000 to $49;999',
                            '    $50;000 to $59;999',
                            '    $60;000 to $69;999',
                            '    $70;000 to $79;999',
                            '    $80;000 and over',
                            '      $80;000 to $89;999',
                            '      $90;000 to $99;999',
                            '      $100;000 and over'
                        ]]
                        df_in['income_div_si'] = diversity.alpha_diversity("simpson", df_dic)
                        df_in['income_div_sh'] = diversity.alpha_diversity("shannon", df_dic)
                        df_in['DAUID'] = da

                        # Pre process Journey to Work
                        df = pd.read_csv(f"{csd}-{da}-Journey%20to%20work.csv", encoding='ISO-8859-1')
                        df = df.loc['Main mode of commuting']['Unnamed: 0']
                        dic = {i[0]: [i[2]] for i in df.index}
                        df_jw = pd.DataFrame.from_dict(dic)
                        df_jw['DAUID'] = da

                        # Pre process Labour
                        df = pd.read_csv(f"{csd}-{da}-Labour.csv", encoding='ISO-8859-1')
                        df_lb = clean(df)
                        n_cols = []
                        for col in df_lb.columns:
                            try: n_cols.append(col.strip())
                            except: n_cols.append(col)
                        df_lb.columns = n_cols
                        df_lb_status = df_lb.iloc[:, 1:9]
                        df_lb_class = df_lb.iloc[:, 15:20]
                        df_lb_place = df_lb.iloc[:, 56:61]
                        df_lb = pd.concat([df_lb_status, df_lb_class, df_lb_place], axis=1)
                        df_lb = df_lb.rename(columns={
                            'Total - Population aged 15 years and over by Labour force status - 25% sample data':'total_labour_force',
                            'Total labour force aged 15 years and over by class of worker - 25% sample data':'total_class_worker',
                            'Total - Place of work status for the employed labour force aged 15 years and over in private households - 25% sample data':'total_place'
                        }).astype(float, errors='ignore')
                        df_lb['worked_home_ratio'] = df_lb['Worked at home']/df_lb['total_place']
                        df_lb['worked_abroad_ratio'] = df_lb['Worked outside Canada']/df_lb['total_place']
                        df_lb['worked_flexible'] = df_lb['No fixed workplace address']/df_lb['total_place']
                        df_lb['worked_usual_ratio'] = df_lb['Worked at usual place']/df_lb['total_place']
                        df_lb['DAUID'] = da

                        # Append data to gdf
                        joi_df = pd.concat([df_ag, df_et, df_ed, df_ho, df_in, df_jw, df_lb], axis=1)
                        out_df = pd.concat([out_df, joi_df], axis=0)

                    # Join data to dissemination areas
                    out_df = out_df.loc[:, ~out_df.columns.duplicated()].reset_index(drop=True)
                    city.DAs = city.DAs.merge(out_df, on='DAUID', suffixes=[None, None])
                    mob_cols = ['  Car; truck; van - as a driver', '  Car; truck; van - as a passenger',
                        '  Public transit', '  Walked', '  Bicycle', '  Other method']
                    mob_df = city.DAs.loc[:, mob_cols]
                    total = mob_df.dropna().astype(int).sum(axis=1)
                    mob_df['DAUID'] = city.DAs.DAUID
                    mob_df = mob_df.dropna().astype(int)

                    # Calculate ratios
                    ratios = pd.DataFrame()
                    ratios['walk'] = mob_df['  Walked'] / total
                    ratios['bike'] = mob_df['  Bicycle'] / total
                    ratios['drive'] = (mob_df['  Car; truck; van - as a driver'] + \
                        mob_df['  Car; truck; van - as a passenger']) / total
                    ratios['bus'] = mob_df['  Public transit'] / total
                    ratios['DAUID'] = mob_df['DAUID'].astype(str)
                    city.DAs = city.DAs.merge(ratios, on='DAUID', suffixes=[None, None], how='outer')

                    # Save it to GeoPackage
                    city.DAs.to_file(city.gpkg, layer='land_dissemination_area', driver='GPKG')
                    print(f'Census dissemination area downloaded and saved at {city.gpkg}')


class Vancouver:
    """Web scraping Vancouver OpenData"""
    def __init__(self, directory):
        self.directory = directory
        self.scraper = Scraper()
        return

    def train_ner(self, blank=False):
        if blank: nlp = spacy.blank('en')
        else: nlp = spacy.load('en_core_web_sm')

        # Getting the pipeline component
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)
        else: ner = nlp.get_pipe('ner')

        # Training data
        train_data = [
            ("2-26 East 1st Avenue is part of CD-1 (464), a Comprehensive Development zone that was approved by Council on June 27, 2006.", {
                "entities": [(0, 20, "LOC"), (32, 42, "CD_ZONE")]}),
            ("a floor space ratio (FSR) of 6.98", {"entities": [(29, 33, "FSR")]}),
            ("a maximum building height of 53.3 m (174.7 ft.); and", {"entities": [(29, 35, "HEIGHT")]}),
            ("169 underground parking spaces.", {"entities": [(0, 4, "PARKING")]}),
            ("20 vehicle parking spaces and 106 bicycle parking spaces", {"entities": [(0, 2, "PARKING")]}),
            ("The zoning would change from RM-4 (multiple dwelling) to CD-1 (comprehensive development).", {
                "entities": [(29, 34, "ZONE"), (57, 62, "CD_ZONE")]}),
            ("On August 3, 2017, the City of Vancouver received an application from Musson Cattell Mackey Partnership Architects to rezone 2715 West 12th Avenue from RS-7 (One-Family Dwelling) District to CD-1 (Comprehensive Development) District, to permit a 3.5-storey townhouse development.", {
                "entities": [(125, 147, "LOC"), (152, 157, "ZONE"), (191, 196, "CD_ZONE")]}),
            ("The City of Vancouver has received a revised application from the Red Door Housing Society to rezone 870 East 8th Avenue from RM-4 (Multiple Dwelling) District to CD-1 (Comprehensive Development) District.", {
                "entities": [(101, 121, "LOC"), (126, 131, "ZONE"), (163, 168, "CD_ZONE")]}),
            ("a building height of 21.3 m (69.9 ft.); ", {"entities": [(21, 27, "HEIGHT")]}),
            ("a floor space ratio (FSR) of 2.80", {"entities": [(29, 33, "FSR")]}),
            ("an overall floor space ratio of 10.9 FSR;", {"entities": [(32, 37, "FSR")]}),
            ("Phone 604-806-0268 ", {"entities": [(6, 19, "PHONE")]}),
            ("a height of 62 ft. (18.9 m)", {"entities": [(12, 18, "HEIGHT"), (20, 26, "HEIGHT")]}),
            ("The proposal is for a 16-storey mixed-use building with ground-level commercial uses and 188 residential units. The proposal includes a floor space ratio (FSR) of 8.86, a  height of 146 ft. (44.5 m) from Main and 152 ft. (46.3 m) from the lane,  72 parking spaces, and 235 bicycle stalls.", {
                "entities": [(163, 167, "FSR"), (183, 190, "HEIGHT"), (191, 197, "HEIGHT"), (246, 249, "PARKING")]}),
            ("The proposal consists of a height of 410.8 ft. (125.22 m), a floor space ratio (FSR) of 8.6", {
                "entities": [(37, 45, "HEIGHT"), (48, 56, "HEIGHT"), (88, 91, "FSR")]})
        ]

        print(train_data)

        # Add labels to 'ner'
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # Disable pipeline components you dont need to change
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

        # Training the model
        with nlp.disable_pipes(*unaffected_pipes):

            # Training for 30 iterations
            for iteration in range(30):
                optimizer = nlp.begin_training()
                # Shuffling examples  before every iteration
                random.shuffle(train_data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        losses=losses,
                        sgd=optimizer
                    )
                    print("Losses", losses)
        return nlp

    def get_url_pdfs(self, run=True, url='https://development.vancouver.ca/'):
        if run:
            # Get list of pdfs from ongoing development permits at City of Vancouver webpage
            links = self.scraper.get_sublinks(url)
            pdfs = {}
            ignore = ['mailto:planninginfo@vancouver.ca']

            # Get pdfs in links and sublinks
            for link in links:
                print(link)
                if (link not in ignore) & (link is not None):
                    if ('.pdf' in link) or ('.PDF' in link) or ('mailto' in link):
                        pass
                    else:
                        try: name = link.split(url)[1]
                        except: name = 'unknown'
                        pdfs[name] = []
                        sublinks = self.scraper.get_sublinks(link)
                        if sublinks is not None:
                            for sublink in sublinks:
                                if sublink is None:
                                    pass
                                elif '.pdf' in sublink:
                                    pdfs[name].append(sublink)

            # Download pdfs and move it to a folder
            for key, value in pdfs.items():
                for pdf in value:
                    if 'file:///' not in pdf:
                        try: pieces = pdf.split(url)[1]
                        except: pieces = pdf
                        pieces = pieces.split('/')
                        filename = '_'.join(pieces)
                        print(f"> Downloading and saving {filename}")
                        response = requests.get(pdf)

                        dir_name = f"{self.directory}{url.split('//')[1].split('.')[0]}/"
                        if not os.path.exists(dir_name): os.mkdir(dir_name)
                        with open(f'{dir_name}{filename}', 'wb') as f:
                            f.write(response.content)
                        print(f"> {filename} downloaded and saved")

            return print(str(len(pdfs)) + ' documents downloaded at ' + str(datetime.datetime.now()))


    def scrape_rezoning(self, run=True):

        # nlp = self.train_ner()
        if run:

            sl = self.scraper.get_sublinks('https://rezoning.vancouver.ca/applications/archives/index.htm')
            applications = pd.DataFrame()
            for i, link in enumerate(sl):
                try:
                    if (len(link.split('/')) > 5) and ('#' not in link):
                        page_id = link.split('/')[len(link.split('/'))-2]
                        applications.at[i, 'ID'] = page_id

                        page = requests.get(link)
                        tree = html.fromstring(page.content)
                        tree.xpath("//h1/text()")

                        h2s = tree.xpath("//h2/text()")
                        lists = tree.xpath("//li/text()")
                        strong = tree.xpath('//span[@class="boldText"]/text()') + tree.xpath('//p[@class="boldText"]/text()') + tree.xpath("//strong/text()")
                        prg = tree.xpath('//p/text()')

                        # Get rezoning information
                        def find_dates(txt):
                            months = [
                                'January', 'February', 'March', 'April', 'May', 'June',
                                'July', 'August', 'September', 'October', 'November', 'December'
                            ]

                            txt = re.sub(' +', ' ', txt)
                            if "\r\n" in txt:
                                txt = txt.replace("\r\n", " ")
                            for m in months:
                                if m in txt:
                                    return txt[txt.find(m):txt.find(m) + len(m) + 9]

                        # Find approval dates
                        approved = []
                        for j, text in enumerate(strong):
                            d = find_dates(text)
                            if 'approved' in text:
                                approved.append(pd.to_datetime(d))
                            elif ('approved' in strong[j - 1]) and (len(approved) == 0):
                                approved.append(pd.to_datetime(d))

                        if len(approved) == 0: pass
                        elif len(approved) > 1:
                            applications.at[i, 'Approved'] = \
                                pd.Series(approved).sort_values(ascending=False).reset_index(drop=True)[0]
                        else:
                            applications.at[i, 'Approved'] = approved[0]

                        # Find application date
                        applied = []
                        for text in h2s:
                            d = find_dates(text)
                            applied.append(pd.to_datetime(d))

                        if len(applied) == 0: pass
                        elif len(applied) > 1:
                            applications.at[i, 'Applied'] = \
                                pd.Series(applied).sort_values(ascending=True).reset_index(drop=True)[0]
                        else:
                            applications.at[i, 'Applied'] = applied[0]

                            # doc = nlp(str(text))
                            # ent_list = [(ent.text, ent.label_) for ent in doc.ents]

                            # if len(ent_list) > 0:
                            #     print(
                            #         f"Paragraph: {text} \n"
                            #         "Entities", ent_list
                            #     )

                        table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")

                        # Get rental housing info
                        full_txt = ' '.join(h2s + strong + lists + prg)
                        applications.at[i, 'Rental'] = [1 if ('rental' in full_txt) or ('Rental' in full_txt) else 0][0]
                except: pass

            app_clean = applications.dropna(subset=['Approved', 'Applied'])
            print(f"> {len(app_clean)} rezoning applications found")
            app_clean = app_clean.sort_values('Approved')

            app_clean['Year'] = [date.year for date in app_clean['Approved']]
            app_clean['Duration'] = app_clean['Approved'] - app_clean['Applied']
            app_clean.to_csv(f'{self.directory}Rezoning.csv', index=False)

            return applications

    def close_session(self):
        return self.scraper.close_session()
