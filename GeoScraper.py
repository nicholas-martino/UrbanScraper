# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:52:23 2019

@author: nicholas-martino

"""

import datetime
import glob
import io
import logging
import os
import time
from io import StringIO
from random import randint, seed
from urllib import request, error
from nltk.chunk import conlltags2tree, tree2conlltags
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import geopandas as gpd
import osmnx as ox
import pandas as pd
import regex
import requests
import selenium.webdriver as webdriver
from Learning.Scraping import Scraper
from Statistics.basic_stats import normalize
from Visualization import polar_chart as pc
from craigslist import CraigslistHousing
import geopy
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from selenium.webdriver.firefox.options import Options
from shapely import affinity
from shapely import wkt
from shapely.geometry import *
from shapely.ops import split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from elementslab.Geospatial import BritishColumbia, Canada
import requests
import bs4
from bs4 import BeautifulSoup


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
                        for i in div1.find_all(name="span", attrs={"class": "icl-u-lg-inline icl-u-xs-hide"}):
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
                    try: geom = Point(locator.geocode(f"{address}")[1])
                    except: geom = 'Unknown'
                data['geometry'].append(geom)

        # Close web browser
        s.driver.close()

        # Export to GeoPackage
        gdf = gpd.GeoDataFrame().from_dict(data)
        gdf = gdf.loc[gdf['geometry'] != 'Unknown']
        gdf.set_geometry(gdf.geometry, drop=True, inplace=True)
        gdf.crs = 4326
        gdf.to_crs(epsg=self.city.crs, inplace=True)
        try:
            gdf0 = gpd.read_file(self.city.gpkg, layer='indeed_employment')
            gdf = pd.concat([gdf0, gdf])
        except: pass
        gdf.drop_duplicates(inplace=True)
        gdf.to_file(self.city.gpkg, layer='indeed_employment')
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


        zillow_id = 'X1-ZWz1hjj0anelu3_211k7'
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
    def air_quality(self, token='306407099d8a8d50a5254d4d0109975aeaf52447'):
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
    def movement_osm_gps(self):
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
                            gdf1 = gdf.read_file(f"{directory}/{file_name}", layer='track_points')
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

    def public_transit_reach(self, time=30):

        return

    def social_twitter(self):
        return self

    def social_instagram(self):
        return self

    # Clean and structure
    def get_gh_gravity(self, radius):
        gdf = gpd.read_file(self.directory + '/' + self.buildings)
        print(gdf)
        for i in radius:
            with open(self.directory + '/' + self.buildings + '_gravity' + str(i) + '.tsv') as tsv:
                print(tsv)

    def layers_from_files(self, file_format='.geojson'):
        files = os.listdir(self.directory)
        file_list = []
        for i in files:
            if i[-(len(file_format)):] == file_format:
                file_list.append(i)
        # Iterate over experiments
        for i in file_list:
            filepath = (self.directory + '/' + i)
            buildings = gpd.read_file(filepath)
            buildings.crs = {'init': 'epsg:26910'}
            c_hull = buildings.unary_union.convex_hull
        return file_list

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


if __name__ == '__main__':
    print(f"Start @ {datetime.datetime.now()}")
    for i in range(100000):

        regions = {
            'Canada':
                {
                    'British Columbia': ['Metro Vancouver']
                }
        }

        for key, value in regions.items():

            # Get data
            bc = BritishColumbia(cities=value['British Columbia'])
            country = Canada(provinces=[bc])
            for city in bc.cities:
                scraper = GeoScraper(city=city)
                scraper.city.update_databases(bound=True, net=True)
                mov = scraper.movement_osm_gps()  # OpenStreetMaps
                emp = scraper.employment_indeed()  # Indeed
                clh = scraper.housing_craigslist('vancouver', 900)  # Craigslist
                aqi = scraper.air_quality()  # Air Quality Index
                print('###')
            bc.update_databases(icbc=True)  # ICBC
            bca_dir = '/Volumes/ELabs/50_projects/16_PICS/07_BCA data/'
            bc.aggregate_bca_from_field(
                inventory_dir=f'{bca_dir}170811_BCA_Provincial_Data/Inventory Information - RY 2017 - Greater Vancouver.csv',
                geodatabase_dir=f'{bca_dir}Juchan_backup/BCA_2017_roll_number_method/BCA_2017_roll_number_method.gdb')
            country.update_databases(census=True)  # StatsCan

            # Analyze data
            for city in bc.cities:
                city.network_analysis(
                    service_areas=[400, 800, 1600],
                    sample_gdf=gpd.read_file(city.gpkg, layer='land_dissemination_area'),
                    aggregated_layers={
                        'network_links': ["network_links_ct"],
                        'network_nodes': ["network_nodes_ct"],
                        'land_assessment_fabric': [
                            "land_assessment_fabric_ct", "NUMBER_OF_BEDROOMS", "NUMBER_OF_BATHROOMS", "elab_use"],
                        'air_quality': ["aqi"],
                        'craigslist_housing': ["craigslist_housing_ct", "price_sqft"],
                        'icbc_accidents': ["icbc_accidents_ct", "Crash Count"]
                    })

        # Scrape development permits at the City of Vancouver webpage
        van = Vancouver('/Users/nicholasmartino/GoogleDrive/Geospatial/Databases/Permits/')
        van.get_permits()

        print(f"Finished @ {datetime.datetime.now()}")
        time.sleep(14400)
