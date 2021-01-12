import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon

def csv2gdf (csv, crs=26910, delimiter=';'):
    df = pd.read_csv(csv, encoding='utf-8', delimiter=delimiter)
    for i, geom in enumerate(df['geometry']):
        try:
            geom.apply(wkt.loads)
        except:
            pass
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.crs = crs
    return gdf

def polygon_grid(gdf, cell_size=10):

    # make shapely polygon grid
    xmin, ymin, xmax, ymax = gdf.total_bounds

    length = cell_size
    width = cell_size

    cols = list(np.arange(np.floor(xmin), np.ceil(xmax), width))
    rows = list(np.arange(np.floor(ymin), np.ceil(ymax), length))
    rows.reverse()

    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(Polygon([(x, y), (x + width, y), (x + width, y - length), (x, y - length)]))

    # make grid_gdf, save to .geojson
    grid_gdf = gpd.GeoDataFrame({'geometry': polygons})
    grid_gdf.crs = "EPSG:26910"
    return grid_gdf

if __name__ == '__main__':
    aggregate = True
    directory = f'/Volumes/Samsung_T5/Databases/CityOpenData'
    gdfs = []
    if directory[:-1] != '/': directory = f'{directory}/'
    for file in os.listdir(directory):
        if '.csv' in file:
            try:
                print(file)
                gdf = csv2gdf(f'{directory}{file}', 26910)
                if not aggregate: gdf.to_file(f'{directory}{file.split(".")[0]}.shp', driver='ESRI Shapefile')
                gdfs.append(gdf)
            except:
                print(f"{file} processing has failed")

    if aggregate:
        final = gpd.GeoDataFrame(pd.merge(pd.concat(gdfs).groupby('index').mean(), gdfs[0].loc[:, ['index', 'geometry']], how='left', on='index'))
        final.crs = 26910
        final.to_file(f'{directory.split("/")[len(directory.split("/"))-1]} - Aggregate.shp', driver='ESRI Shapefile')
