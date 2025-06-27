from datetime import datetime
import math
import asyncio
from functools import cache
from typing import Tuple, TypedDict
import io
import json

import requests
import xmltodict
import aiohttp
from PIL import Image


class WaybackLayer(TypedDict):
    identifier: str
    layer_number: int
    publish_date: str # formatted YYYY-MM-DD


class WaybackImageLayer(WaybackLayer):
    image: Image.Image
    approximate_acquisition_date: str # formatted YYYY-MM-DD
    point_pixel_offset_xy: Tuple[float, float]


def parse_layer(layer) -> WaybackLayer:
    parsed_layer = WaybackLayer(
        publish_date=layer["ows:Title"][-11:-1],
        layer_number=int(layer["ResourceURL"]["@template"].split("/")[-4]),
        identifier=layer["ows:Identifier"].split("_", 1)[-1].lower(),
    )
    return parsed_layer


@cache
def get_wayback_layers() -> list[WaybackLayer]:
    """Returns a dictionary with layer_ids as keys and their associated dates as values"""
    capabilities = requests.get(
        "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml"
    )
    layers = xmltodict.parse(capabilities.text)["Capabilities"]["Contents"]["Layer"]
    layer_data = [parse_layer(layer) for layer in layers]
    sorted_layer_data = sorted(
        layer_data, key=lambda d: d["publish_date"], reverse=True
    )
    return sorted_layer_data


def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert latitude and longitude to XYZ slippy tile coordinates at a given zoom level."""
    # Convert latitude to radians
    lat_rad = math.radians(lat)

    # Compute tile coordinates
    n = 2.0**zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    return x_tile, y_tile


def get_available_layer_ids(lat: float, lon: float, zoom: int) -> list[WaybackLayer]:
    """
    This goes chronologically through the layers in the wayback archive
    and selects all layers where data is available at the chosen point

    Logic is taken from here: https://github.com/vannizhang/wayback-core/blob/246910537a2f33a5359b48d56ab1a1ba739c8a69/src/change-detector/index.ts#L58

    """
    layers = get_wayback_layers()
    layer_keys = [layer["layer_number"] for layer in layers]
    row, col = latlon_to_tile(lat, lon, zoom)
    data_available = 1
    layer_index = 0
    available_layers = []
    while data_available == 1 and layer_index < len(layer_keys):
        current_layer = layers[layer_index]
        layer_info = requests.get(
            f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tilemap/{current_layer['layer_number']}/{zoom}/{col}/{row}"
        ).json()
        data_available = layer_info["data"][0]
        if data_available == 1:
            layer_number = layer_info.get("select", [current_layer["layer_number"]])[0]
            layer_index = layer_keys.index(layer_number) + 1
            available_layers.append(layers[layer_index - 1])
    return available_layers


# --- Asynchronous tile fetch with retries ---
async def fetch_tile(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> Image.Image | None:
    """
    Asynchronously fetch a tile image using aiohttp, retrying on failure.
    """
    for attempt in range(retries):
        # Acquire the semaphore to limit concurrency.
        async with semaphore:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.read()
                    # Open the image from bytes and convert to RGBA.
                    tile_img = Image.open(io.BytesIO(data)).convert("RGBA")
                    return tile_img
            except Exception as e:
                if attempt < retries - 1:
                    # Wait a moment before retrying.
                    await asyncio.sleep(1)
                else:
                    print(
                        f"Warning: Could not load tile from {url} after {retries} attempts. Error: {e}"
                    )
                    return None


def latlon_to_web_mercator(lat: float, lon: float) -> Tuple[float, float]:
    """
    Converts lat/lon in WGS84 to Web Mercator (EPSG:3857) coordinates.
    """
    # The radius of the Earth in meters for Web Mercator.
    # More precisely, the projection is defined for a sphere with radius 6378137.
    origin_shift = 20037508.342789244  # half the Earth's circumference in meters
    x = lon * origin_shift / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) * (origin_shift / math.pi)
    return x, y


def latlon_to_tile_pixel(
    lat: float, lon: float, zoom: int, tile_size: int = 256
) -> Tuple[float, float]:
    """
    Given a point specified by latitude and longitude (in degrees) and a zoom level,
    returns the tile (column, row) that covers the point and the pixel
    coordinates (px, py) within that tile.
    """
    # Convert lat/lon to Web Mercator coordinates (meters)
    x, y = latlon_to_web_mercator(lat, lon)

    # Calculate the total map size in pixels at this zoom level.
    map_size = tile_size * (2**zoom)

    # Web Mercator spans from -origin_shift to +origin_shift.
    origin_shift = 20037508.342789244

    # Convert the point to global pixel coordinates.
    global_pixel_x = ((x + origin_shift) / (2 * origin_shift)) * map_size
    global_pixel_y = ((origin_shift - y) / (2 * origin_shift)) * map_size

    # Determine the tile indices (column and row) by integer division.
    tile_col = int(global_pixel_x // tile_size)
    tile_row = int(global_pixel_y // tile_size)

    # Calculate the pixel coordinates within the tile.
    pixel_in_tile_x = global_pixel_x - (tile_col * tile_size)
    pixel_in_tile_y = global_pixel_y - (tile_row * tile_size)

    return pixel_in_tile_x, pixel_in_tile_y


async def to_image(
    lat: float,
    lon: float,
    zoom: int,
    xyz_url: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    number_tiles: int = 3,
) -> Image.Image:
    # Convert lat/lon to global pixel coordinates
    x, y = latlon_to_tile(lat, lon, zoom)
    start_x = x - 1
    start_y = y - 1

    # Create a blank image to stitch tiles into
    stitched = Image.new("RGB", (number_tiles * 256, number_tiles * 256))

    # Create tasks for all required tile fetches.
    tasks = []
    coords = []  # To keep track of which tile belongs where.

    # Fetch and stitch tiles
    for tx in range(start_x, start_x + number_tiles):
        for ty in range(start_y, start_y + number_tiles):
            # Fetch tile image
            url = xyz_url.format(z=zoom, x=tx, y=ty)
            # Calculate position in stitched image
            pos_x = (tx - start_x) * 256
            pos_y = (ty - start_y) * 256
            tasks.append(fetch_tile(session, url, semaphore, retries=3))
            coords.append((pos_x, pos_y))

        # Run all tile requests concurrently.
    results = await asyncio.gather(*tasks)

    # Paste each successfully fetched tile into the full_image.
    for (x, y), tile_img in zip(coords, results):
        if tile_img is not None:
            stitched.paste(tile_img, (x, y))

    return stitched


async def get_acquisition_date(
    lat: float, lon: float, identifier: str, session: aiohttp.ClientSession
) -> str:
    params = {
        "f": "json",
        "where": "1=1",
        "outFields": "SRC_DATE2",
        "geometry": json.dumps(
            {"spatialReference": {"wkid": 4326}, "x": lon, "y": lat}
        ),
        "returnGeometry": "false",
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
    }
    metadata_template = f"https://metadata.maptiles.arcgis.com/arcgis/rest/services/World_Imagery_Metadata_{identifier}/MapServer/6/query"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    async with session.get(
        metadata_template, params=params, headers=headers
    ) as response:
        response.raise_for_status()
        data = await response.json(content_type=None)
        try:
            epoch = data["features"][0]["attributes"]["SRC_DATE2"]
            publish_date = datetime.fromtimestamp(epoch / 1000)  # from ms to s since epoch
            publish_date_str = publish_date.strftime('%Y-%m-%d')
        except IndexError:
            publish_date_str = 'unknown'
    return publish_date_str


async def get_vhr(lat, lon, zoom, start, start_date: str = '1900-01-01', remove_duplicates: bool = False):
    # temporary shim without network calls 
    import pickle
    with open("test/vhr_example.pickle", "rb") as f:
        vhr_data = pickle.load(f)
    return vhr_data

# async def get_vhr(lat: float, lon: float, zoom: int, start_date: str = '1900-01-01', remove_duplicates: bool = False) -> list[WaybackImageLayer]:
#     number_tiles = 3
#     available_layers = get_available_layer_ids(
#         lat, lon, zoom
#     )  # choosing slightly lower zoom, to get more available layers
#     # Set up an aiohttp session and a semaphore (limit to 10 concurrent requests).
#     async with aiohttp.ClientSession() as session:
#         semaphore = asyncio.Semaphore(10)
#         tasks = []
#         metadata_tasks = []
#         for layer in available_layers:
#             # Build the URL template for this layer.
#             url_template = (
#                 f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/MapServer/tile/"
#                 f"{layer['layer_number']}/{{z}}/{{y}}/{{x}}"
#             )
#             metadata_tasks.append(
#                 get_acquisition_date(lat, lon, layer["identifier"], session)
#             )

#         # Run all layer tasks concurrently.
#         acquisition_dates = await asyncio.gather(*metadata_tasks)
#         acquisitions_layers_sorted = sorted(zip(acquisition_dates, available_layers), reverse=True, key=lambda x: x[0])
#         acquisitions_layers = []
#         unique_acquisitions = set()
#         for acquisition_date, layer in acquisitions_layers_sorted:
#             if acquisition_date < start_date:
#                 continue
#             if (acquisition_date in unique_acquisitions) and remove_duplicates:
#                 continue   
#             unique_acquisitions.add(acquisition_date)
#             acquisitions_layers.append((acquisition_date, layer))
        

#         for _, layer in acquisitions_layers:
#             # Build the URL template for this layer.
#             url_template = (
#                 f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/MapServer/tile/"
#                 f"{layer['layer_number']}/{{z}}/{{y}}/{{x}}"
#             )
#             tasks.append(
#                 to_image(lat, lon, zoom, url_template, session, semaphore, number_tiles)
#             )
#         images = await asyncio.gather(*tasks)

#     offset_x, offset_y = latlon_to_tile_pixel(lat, lon, zoom)

#     layers_with_images = []
#     for (acquisition_date, layer), image in zip(acquisitions_layers, images):
#         layers_with_images.append(
#             WaybackImageLayer(
#                 approximate_acquisition_date=acquisition_date,
#                 image=image,
#                 point_pixel_offset_xy=(
#                     offset_x + (number_tiles // 2) * 256,
#                     offset_y + (number_tiles // 2) * 256,
#                 ),
#                 **layer,
#             )
#         )
#     # Return a dummy image, if nothing is returned
#     if len(layers_with_images) == 0:
#          layers_with_images.append(
#             WaybackImageLayer(
#                 approximate_acquisition_date='no data',
#                 publish_date='no data',
#                 image=Image.new("RGB", (number_tiles*256, number_tiles*256)),
#                 point_pixel_offset_xy=(
#                     offset_x + (number_tiles // 2) * 256,
#                     offset_y + (number_tiles // 2) * 256,
#                 ),
#                 identifier="None",
#                 layer_number=0
#             )
#         )
#     return layers_with_images


if __name__ == "__main__":
    lat, lon = 47.071278, 15.4347115
    zoom = 17
    layers = asyncio.run(get_vhr(lat, lon, zoom))
