import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.morphology import closing, disk, skeletonize
from skimage.measure import label, regionprops_table

import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape 
import rasterio

gdal.UseExceptions()

def read_georaster(path):
    dataset: gdal.DataSet = gdal.Open(path)
    count = dataset.RasterCount
    print("Liczba kanałów:", count)

    bands = [dataset.GetRasterBand(i+1).ReadAsArray() for i in range(count)]
    return np.dstack(bands).astype(np.float32)

def select_dark_nir(bands, nir=7, percentile=10): # zostawiamy tylko 10% najciemniejszych pikseli w nir
    nir_band = bands[..., nir].astype(float)

    valid_mask = nir_band != 0
    valid_vals = nir_band[valid_mask]

    threshold = np.percentile(valid_vals, percentile)

    print(f"Zostawiamy tylko piksele o wyższej wartości niż: {threshold:.3f}")

    dark_mask = (nir_band <= threshold) & valid_mask

    return dark_mask

def calc_ndwi(bands, green=3, nir=7):
    green = bands[..., green]
    nir = bands[..., nir]

    ndwi = (green - nir) / (green + nir + 1e-6)
    return ndwi

def filter_river(ndwi, 
                 threshold=-0.55, # próg NDWI, powyżej woda
                 disk_size = 4, # będziemy "zaklejać" obiekty, tworząc bufor o podanej pikseli wokół każdego piksela
                 min_area=300, # mniejsze obiekty będą wywalane, żeby wywalić szumy i zmniejszyć ilość obiektów
                 elong_threshold=500): # wskaznik wydłużenia
    
    water_mask = ndwi > threshold

    # uzupełniamy rzeki (dylatacja + erozja)
    # najpierw rozszerza każdy piksel o liczbę pikseli okresloną w disk, żeby 
    # połączyć poprzerywane obiekty, a później obcina brzegi, żeby się rzeka nie pogrubiła.
    # Robimy to, żeby zmniejszyć ilość obiektów (plam), bo pozniej dla każdego obiektu będziemy liczyć obwód i powierzchnię,
    # a im mniej obiektów tym mniej iteracji + wstępnie połączymy dziury (chociaz trzeba to będzie jeszcze potem powtórzyć)
    # chociaż to też chwilę trwa niestety...
    water_mask = closing(water_mask, disk(disk_size))

    labeled = label(water_mask) # każdemu wykrytemu obiektowi (plamie pikseli) przyporządkowujemy etykietę

    table = regionprops_table(
        labeled,
        properties=("label", "area", "perimeter")
    ) # generujemy tabelę, w której trzymamy dla każdego obiektu powierzchnię i obwód
    
    # współczynnik wydłużenia (im większy, tym bardziej "linia" niż "plama")
    elong = (table["perimeter"]**2) / (table["area"]) # (obwod^2/powierzchnia)
    good = (
        (table["area"] > min_area) & # odrzucam pojedyncze piksele małe
        (elong > elong_threshold)  #  odrzucam niewydłużone obiekty 
    )
    
    good_labels = table["label"][good]
    river_mask = np.isin(labeled, good_labels)

    #==== druga iteracja : 

    water_mask2 = closing(river_mask, disk(10)) 
    labeled2 = label(water_mask2)

    table2 = regionprops_table(
        labeled2,
        properties=("label", "area", "perimeter")
    )

    elong2 = (table2["perimeter"]**2) / (table2["area"]) # (obwod^2/powierzchnia)
    good2 = (
        (table2["area"] > min_area) & # odrzucam pojedyncze piksele małe
        (elong2 > elong_threshold)  # odrzucam niewydłużone obiekty
    )

    final_labels = table2["label"][good2]
    final_river_mask = np.isin(labeled2, final_labels)

    plt.figure(figsize=(8,5))
    plt.imshow(final_river_mask, cmap="Blues")
    plt.title("Wykryta rzeka")
    plt.show()

    percent = np.count_nonzero(river_mask) / ndwi.size * 100
    print(f"Zostawiono {percent:.2f}% pikseli")
    print(f"Znaleziono {len(good_labels)} obiektów spełniających kryterium rzeki")

    return final_river_mask

def save_mask_to_shp(mask, src_path, out_shp = "wykryta_rzeka.shp" ):

    with rasterio.open(src_path) as src:
        transform = src.transform # do konwersji piskeli na xy
        crs = src.crs 

    mask_bool = mask.astype(np.uint8) #flase/true->0/1 

    polygons = []
    for geom, val in shapes(mask_bool, transform=transform):
        if val == 1:
            polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame(
        {"id": range(len(polygons))},
        geometry=polygons,
        crs=crs   
    )

    gdf.to_file(out_shp)

    return gdf

if __name__ == "__main__": 

    raster = r"grupa_12.tif"
    bands = read_georaster(raster)

    dark_mask = select_dark_nir(bands, nir=7, percentile=10)

    ndwi_mask = calc_ndwi(bands, green=3, nir=7)

    ndwi = np.where(dark_mask, ndwi_mask, np.nan)

    river = filter_river(ndwi)

    result = save_mask_to_shp(river, raster)
