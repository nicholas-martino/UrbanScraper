import os
from zipfile import ZipFile

import geopandas as gpd
import requests
from tqdm import tqdm


dbf = gpd.read_file('data/lidar-2013.dbf')
out = '/Volumes/ELabs/50_projects/19_SSHRC_Insight/urban forestry/Trees/lidar'

for url in tqdm(dbf['lidar_url']):
	r = requests.get(url, stream=True)
	file_dir = f"{out}/{url.split('/')[len(url.split('/')) - 1]}"
	with open(file_dir, 'wb') as file:
		for chunk in r.iter_content(chunk_size=128):
			file.write(chunk)
	with ZipFile(file_dir, 'r') as zip_ref:
		zip_ref.extractall(out)
	os.remove(file_dir)
