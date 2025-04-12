import pandas as pd
from pyproj import Transformer
from geopy.geocoders import Nominatim
from typing import Dict, Tuple, Optional

# Cache for country UTM zones to avoid repeated geocoding
country_zone_cache: Dict[str, int] = {}

def get_utm_zone_from_country(country: str) -> Optional[int]:
    """Get UTM zone from country name using geocoding with caching."""
    if country in country_zone_cache:
        return country_zone_cache[country]
    
    try:
        geolocator = Nominatim(user_agent="birdnet_analyzer")
        location = geolocator.geocode(country)
        if location:
            lon = location.longitude
            zone = int((lon + 180) / 6) + 1
            country_zone_cache[country] = zone
            return zone
    except Exception:
        return None
    return None

def determine_hemisphere(latitude: float) -> bool:
    """Determine if coordinate is in northern hemisphere."""
    return latitude >= 0

def process_coordinates(df: pd.DataFrame, 
                       country_col: str,
                       easting_col: str,
                       northing_col: str) -> pd.DataFrame:
    """Process UTM coordinates in dataframe to get lat/lon with country caching."""
    # Get unique countries and their zones first
    unique_countries = df[country_col].unique()
    for country in unique_countries:
        if country not in country_zone_cache:
            get_utm_zone_from_country(country)
    
    def convert_row(row):
        zone = country_zone_cache.get(row[country_col])
        if zone is None:
            return pd.Series([None, None])
            
        try:
            # First convert with northern hemisphere assumption to get approximate latitude
            transformer_north = Transformer.from_crs(f"epsg:326{int(zone):02d}", "epsg:4326", always_xy=True)
            test_lon, test_lat = transformer_north.transform(row[easting_col], row[northing_col])
            
            # Now that we know the latitude, we can determine the correct hemisphere
            hemisphere = "326" if determine_hemisphere(test_lat) else "327"
            
            # If hemisphere changed, convert again with correct hemisphere
            if hemisphere == "327":
                transformer = Transformer.from_crs(f"epsg:327{int(zone):02d}", "epsg:4326", always_xy=True)
                lon, lat = transformer.transform(row[easting_col], row[northing_col])
                return pd.Series([lat, lon])
            
            return pd.Series([test_lat, test_lon])
        except Exception:
            return pd.Series([None, None])
    
    # Convert coordinates using the cached zones
    df[['latitude', 'longitude']] = df.apply(convert_row, axis=1)
    return df
