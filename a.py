import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import xml.etree.ElementTree as ET
from geopy.distance import geodesic, distance
from geopy import Point
import requests
import os
import time
import pickle
import datetime as dt
import re
import io
import uuid

# Initialize session state variables
if 'plan_generated' not in st.session_state:
    st.session_state.plan_generated = False
if 'path_msl' not in st.session_state:
    st.session_state.path_msl = None
if 'trigger_points_msl' not in st.session_state:
    st.session_state.trigger_points_msl = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'plan_json' not in st.session_state:
    st.session_state.plan_json = None
if 'expected_photo_count' not in st.session_state:
    st.session_state.expected_photo_count = None
if 'home_pt' not in st.session_state:
    st.session_state.home_pt = None
if 'rtl_pt' not in st.session_state:
    st.session_state.rtl_pt = None
if 'use_home_for_rtl' not in st.session_state:
    st.session_state.use_home_for_rtl = None
if 'home_elev' not in st.session_state:
    st.session_state.home_elev = None
if 'camera' not in st.session_state:
    st.session_state.camera = None

# Global variable to store home elevation
home_elev = None

# -----------------------------------------------------------------------------
# ⛰  BACK‑END HELPERS
# -----------------------------------------------------------------------------

def parse_kml(kml_file):
    """Return the ordered list of (lat, lon) tuples contained in the first
    <LineString> encountered in the KML file."""
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        for ls in root.findall(".//kml:LineString", ns):
            coords = []
            coord_text = ls.find("kml:coordinates", ns).text.strip().split()
            # skip the first pair – it is repeated at the end in many KML exports
            for c in coord_text[1:]:
                lon, lat, *_ = map(float, c.split(","))
                coords.append((lat, lon))
            return coords
        return []
    except Exception as e:
        st.error(f"Error parsing KML: {e}")
        return []

def parse_kml_alt_rel(kml_file):
    """Return the Alt Rel value from the Placemark with Index: 1 in the KML file."""
    try:
        # Read file content to memory to avoid pointer issues
        if hasattr(kml_file, 'seek'):
            kml_file.seek(0)
        content = kml_file.read()
        if not content:
            st.error("KML file is empty.")
            return None
        # Parse as string to handle Streamlit UploadedFile
        tree = ET.parse(io.StringIO(content.decode('utf-8') if isinstance(content, bytes) else content))
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        
        # Extract Alt Rel from the Placemark with Index: 1
        for placemark in root.findall(".//kml:Placemark", ns):
            desc = placemark.find("kml:description", ns)
            if desc is not None and desc.text:
                if "Index: 1" in desc.text:
                    match = re.search(r"Alt Rel: ([\d.]+) m", desc.text)
                    if match:
                        return float(match.group(1))
        st.warning("No Placemark with Index: 1 and Alt Rel found in KML file.")
        return None
    except ET.ParseError as e:
        st.error(f"Error parsing KML for Alt Rel: Invalid XML - {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error parsing KML for Alt Rel: {str(e)}")
        return None

def parse_kml_with_alt_rel(kml_file):
    """Return the ordered list of (lat, lon) tuples from the first LineString
    and the Alt Rel from the Placemark with Index: 1 in the KML file."""
    try:
        # Read file content
        if hasattr(kml_file, 'seek'):
            kml_file.seek(0)
        content = kml_file.read()
        if not content:
            st.error("KML file is empty.")
            return [], None
        # Parse as string
        tree = ET.parse(io.StringIO(content.decode('utf-8') if isinstance(content, bytes) else content))
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        
        # Extract Alt Rel from the Placemark with Index: 1
        alt_rel = None
        for placemark in root.findall(".//kml:Placemark", ns):
            desc = placemark.find("kml:description", ns)
            if desc is not None and desc.text:
                if "Index: 1" in desc.text:
                    match = re.search(r"Alt Rel: ([\d.]+) m", desc.text)
                    if match:
                        alt_rel = float(match.group(1))
                        break
        
        # Parse coordinates from LineString (same as original parse_kml)
        for ls in root.findall(".//kml:LineString", ns):
            coords = []
            coord_text = ls.find("kml:coordinates", ns)
            if coord_text is None or not coord_text.text:
                st.error("No coordinates found in LineString.")
                return [], alt_rel
            coord_text = coord_text.text.strip().split()
            for c in coord_text[1:]:
                lon, lat, *_ = map(float, c.split(","))
                coords.append((lat, lon))
            return coords, alt_rel
        st.warning("No LineString found in KML file.")
        return [], alt_rel
    except ET.ParseError as e:
        st.error(f"Error parsing KML: Invalid XML - {str(e)}")
        return [], None
    except Exception as e:
        st.error(f"Error parsing KML: {str(e)}")
        return [], None

def calculate_angle(p1, p2, p3):
    """Return the turn angle at p2 given three points in deg."""
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 180.0
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1)))

def split_lines_by_turn(coords, thresh=170):
    """Split a poly‑line wherever the turn angle becomes sharper than thresh."""
    segments, cur = [], [coords[0]]
    for i in range(1, len(coords) - 1):
        cur.append(coords[i])
        if calculate_angle(coords[i - 1], coords[i], coords[i + 1]) < thresh:
            segments.append(cur)
            cur = [coords[i]]
    cur.append(coords[-1])
    segments.append(cur)
    return segments

def filter_main_lines(lines, min_len=500):
    """Keep only segments longer than min_len metres."""
    out = []
    for seg in lines:
        d = sum(geodesic(seg[i], seg[i + 1]).meters for i in range(len(seg) - 1))
        if d >= min_len and len(seg) >= 2:
            out.append(seg)
    return out

def adjust_line_directions(lines):
    """Orient all segments so that they can be flown in sequence without
    dog‑legging back to the previous end."""
    if len(lines) > 1:
        for i in range(1, len(lines)):
            prev_end = lines[i - 1][-1]
            curr_start = lines[i][0]
            curr_end = lines[i][-1]
            if geodesic(prev_end, curr_start).meters > geodesic(prev_end, curr_end).meters:
                lines[i].reverse()
    return lines

# -----------------------------------------------------------------------------
# 📡  TERRAIN & WEATHER APIS WITH 24 h CACHE FOR ELEVATION
# -----------------------------------------------------------------------------

# def fetch_elevations(coords, cache_file="elevations_cache.pkl", api_key="YOUR_API_KEY"):
#     cache = {}
#     current_time = time.time()
#     expiration_seconds = 24 * 60 * 60

#     try:
#         with open(cache_file, "rb") as f:
#             cached_data = pickle.load(f)
#             cache_time = cached_data.get("timestamp", 0)
#             if current_time - cache_time <= expiration_seconds:
#                 cache = cached_data.get("elevations", {})
#     except FileNotFoundError:
#         pass

#     # Check for None or invalid elevations
#     uncached_coords = [c for c in coords if c not in cache or cache[c] is None or cache[c] <= 0]
#     if uncached_coords:
#         base_url = "https://maps.googleapis.com/maps/api/elevation/json"
#         for i in range(0, len(uncached_coords), 100):
#             batch = uncached_coords[i : i + 100]
#             # Format locations as lat,lng|lat,lng for Google Maps API
#             locations = "|".join([f"{lat},{lon}" for lat, lon in batch])
#             params = {
#                 "locations": locations,
#                 "key": api_key
#             }
#             try:
#                 r = requests.get(base_url, params=params, timeout=10)
#                 r.raise_for_status()
#                 results = r.json().get("results", [])
#                 elevs = [pt["elevation"] for pt in results]
#                 for coord, elev in zip(batch, elevs):
#                     if elev <= 0:  # Warn about suspicious elevations
#                         st.warning(f"Suspicious elevation {elev} m for coord {coord}. Retrying next run.")
#                         cache[coord] = None  # Mark for retry
#                     else:
#                         cache[coord] = elev
#             except requests.RequestException as e:
#                 st.warning(f"Failed to fetch elevations for batch {i // 100 + 1}: {e}. Marking for retry.")
#                 for coord in batch:
#                     cache[coord] = None  # Mark for retry
#             time.sleep(0.3)

#         with open(cache_file, "wb") as f:
#             pickle.dump({"timestamp": current_time, "elevations": cache}, f)

def fetch_elevations(coords, cache_file="elevations_cache.pkl"):
    cache = {}
    current_time = time.time()
    expiration_seconds = 24 * 60 * 60

    try:
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            cache_time = cached_data.get("timestamp", 0)
            if current_time - cache_time <= expiration_seconds:
                cache = cached_data.get("elevations", {})
    except FileNotFoundError:
        pass

    # Modified: Explicitly check for None or invalid elevations
    uncached_coords = [c for c in coords if c not in cache or cache[c] is None or cache[c] <= 0]
    if uncached_coords:
        url = "https://api.open-elevation.com/api/v1/lookup"
        for i in range(0, len(uncached_coords), 100):
            batch = uncached_coords[i : i + 100]
            locs = [{"latitude": lat, "longitude": lon} for lat, lon in batch]
            try:
                r = requests.post(url, json={"locations": locs}, timeout=10)
                r.raise_for_status()
                results = r.json().get("results", [])
                elevs = [pt["elevation"] for pt in results]
                for coord, elev in zip(batch, elevs):
                    if elev <= 0:  # Warn about suspicious elevations
                        st.warning(f"Suspicious elevation {elev} m for coord {coord}. Retrying next run.")
                        cache[coord] = None  # Mark for retry
                    else:
                        cache[coord] = elev
            except requests.RequestException as e:
                st.warning(f"Failed to fetch elevations for batch {i // 100 + 1}: {e}. Marking for retry.")
                for coord in batch:
                    cache[coord] = None  # Mark for retry
            time.sleep(0.3)

        with open(cache_file, "wb") as f:
            pickle.dump({"timestamp": current_time, "elevations": cache}, f)

    # Return None for invalid elevations to handle in calling function
    return [cache.get(coord, None) if cache.get(coord) is not None and cache.get(coord) > 0 else None for coord in coords]

    # Return None for invalid elevations to handle in calling function
    return [cache.get(coord, None) if cache.get(coord) is not None and cache.get(coord) > 0 else None for coord in coords]

def fetch_weather_data(lat, lon, api_key="8c8ddc6600e68fa4571aaebfe32eca55"):
    current_url = (
        f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    forecast_url = (
        f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    try:
        current_response = requests.get(current_url).json()
        forecast_response = requests.get(forecast_url).json()
        if current_response.get("cod") != 200 or forecast_response.get("cod") != "200":
            return None
        current_weather = current_response
        current = {
            "temperature": current_weather["main"]["temp"],
            "description": current_weather["weather"][0]["description"],
            "wind_speed": current_weather["wind"]["speed"],
            "wind_deg": current_weather["wind"]["deg"],
            "pressure": current_weather["main"]["pressure"],
            "precipitation": current_weather.get("rain", {}).get("1h", 0)
            + current_weather.get("snow", {}).get("1h", 0),
            "humidity": current_weather["main"]["humidity"],
        }
        current_dt = dt.datetime.fromtimestamp(current_weather["dt"], dt.UTC)
        today = current_dt.date()
        dates = [today + dt.timedelta(days=i) for i in range(3)]
        forecast_by_day = {}
        for fc in forecast_response["list"]:
            fc_dt = dt.datetime.fromtimestamp(fc["dt"], dt.UTC)
            fc_date = fc_dt.date()
            if fc_date in dates:
                forecast_by_day.setdefault(fc_date, []).append(fc)
        forecast_list = []
        for day in dates:
            if day in forecast_by_day:
                forecasts = forecast_by_day[day]
                temp_min = min(fc["main"]["temp"] for fc in forecasts)
                temp_max = max(fc["main"]["temp"] for fc in forecasts)
                pop = max(fc["pop"] for fc in forecasts) * 100
                midday_fc = min(
                    forecasts,
                    key=lambda x: abs(dt.datetime.fromtimestamp(x["dt"], dt.UTC).hour - 12),
                )
                description = midday_fc["weather"][0]["description"]
                forecast_list.append(
                    {
                        "date": day.strftime("%Y-%m-%d"),
                        "temp_min": temp_min,
                        "temp_max": temp_max,
                        "description": description,
                        "pop": pop,
                    }
                )
        short_term_pop = forecast_response["list"][0]["pop"] * 100 if forecast_response["list"] else 0
        return {"current": current, "forecast": forecast_list, "short_term_pop": short_term_pop}
    except requests.RequestException:
        return None

# -----------------------------------------------------------------------------
# 🔀  SMALL UTILS
# -----------------------------------------------------------------------------

def dms_to_decimal(dms_str, direction):
    import re

    try:
        pattern = r"^(\d{1,3})°(\d{1,2})'(\d{1,2}(?:\.\d+)?)\"$"
        match = re.match(pattern, dms_str.strip())
        if not match:
            raise ValueError("DMS format must be like '18°26'40.72\"'")
        deg, minu, sec = map(float, match.groups())
        if direction not in ["N", "S", "E", "W"]:
            raise ValueError("Direction must be N, S, E, or W")
        if deg < 0 or minu >= 60 or sec >= 60:
            raise ValueError("Invalid DMS values")
        decimal = deg + minu / 60 + sec / 3600
        if direction in ["S", "W"]:
            decimal = -decimal
        return decimal
    except Exception as e:
        st.error(f"Invalid DMS format: {e}. Use format like '18°26'40.72\"' and select the direction.")
        return None

def deg_to_cardinal(deg):
    dirs = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    ix = int((deg + 11.25) / 22.5)
    return dirs[ix % 16]

def calculate_bearing(p1, p2):
    lat1, lat2 = np.radians(p1[0]), np.radians(p2[0])
    dlon = np.radians(p2[1] - p1[1])
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def move_point_along_bearing(start, bearing, dist_m):
    origin = Point(start[0], start[1])
    dest = distance(meters=dist_m).destination(origin, bearing)
    return (dest.latitude, dest.longitude)

def get_coord_at_distance(path, distances, D):
    """
    Calculate the coordinate (lat, lon) at a given distance D along the path.
    
    Args:
        path: List of (lat, lon, alt) tuples representing the flight path.
        distances: List of cumulative distances along the path (in meters).
        D: Distance along the path (in meters) to find the coordinate for.
    
    Returns:
        Tuple (lat, lon) at distance D.
    """
    i = np.searchsorted(distances, D, side="left")
    if i == len(distances):
        return path[-1][:2]  # Return last point if D exceeds path length
    elif distances[i] == D or i == 0:
        return path[i][:2]   # Return exact waypoint if D matches or is at start
    else:
        j = i - 1
        f = (D - distances[j]) / (distances[i] - distances[j])  # Interpolation factor
        lat = path[j][0] + f * (path[i][0] - path[j][0])
        lon = path[j][1] + f * (path[i][1] - path[j][1])
        return (lat, lon)   

# -----------------------------------------------------------------------------
# ✈  PATH GENERATOR –  *RELATIVE* ALTITUDES (Option A)
# -----------------------------------------------------------------------------

def create_trigger_item(lat, lon, alt_rel, trigger_type, trigger_distance, item_id):
    """Return a Mission item dict for camera trigger if applicable."""
    if trigger_type == "camera":
        return {
            "AMSLAltAboveTerrain": None,
            "Altitude": alt_rel,
            "AltitudeMode": 1,  # relative
            "autoContinue": True,
            "command": 206,  # DO_DIGICAM_CONTROL_DISTANCE
            "doJumpId": item_id,
            "frame": 3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
            "params": [trigger_distance, 0, 1, 1, 0, 0, 0],
            "type": "SimpleItem",
        }
    return None

def is_unusual_terrain(elevs):
    """Check if terrain elevations are unusual (None, 0, negative, or out of range)."""
    if not elevs:
        return True
    valid_elevs = [e for e in elevs if e is not None]
    if not valid_elevs:
        return True
    for e in valid_elevs:
        if e <= 0 or e < -500 or e > 9000:
            return True
    return False

def calculate_flight_height(sensor_width_mm, focal_length_mm, gsd_cm_per_px, image_width_px):
    """
    Calculates the Flight Height (altitude) for a desired Ground Sampling Distance (GSD).

    Args:
        sensor_width_mm (float): The width of the camera sensor in millimeters.
        focal_length_mm (float): The focal length of the lens in millimeters.
        gsd_cm_per_px (float): The desired Ground Sampling Distance in centimeters per pixel.
        image_width_px (int): The width of the image in pixels.

    Returns:
        float: The flight height in meters.
    """
    # Convert units to meters for consistency
    sensor_width_m = sensor_width_mm / 1000
    focal_length_m = focal_length_mm / 1000
    gsd_m_per_px = gsd_cm_per_px / 100

    # Calculate flight height in meters
    flight_height_m = (gsd_m_per_px * focal_length_m * image_width_px) / sensor_width_m

    return flight_height_m

def generate_simplified_path(
    lines,
    home_pt,
    rtl_pt,
    home_elev,
    home_alt=40,
    safety_margin=0,
    turning_length=250,
    trigger_distance=40,
    end_trigger_distance=0,
    kml_alt_rel=100,
    camera=None,
    progress_callback=None
):
    path = []
    trigger_points = []
    item_id_counter = 1
    expected_photo_count = len(lines) * 2
    FALLBACK_SAFETY_MARGIN = 150

    total_steps = len(lines) + 6  # Survey lines + takeoff + loiter + RTL + landing
    current_step = 0

    if progress_callback:
        progress_callback(current_step / total_steps, "Initializing flight path...")

    # Camera parameters
    if camera is None:
        camera = {
            "sensor_height_mm": 15.6,
            "sensor_width_mm": 23.5,
            "focal_length_mm": 16.0,
            "image_height_px": 6000,
            "image_width_px": 4000,
        }

    # Calculate required AGL altitude for GSD = 3.2 cm/pixel
    target_gsd = 3.2
    required_alt_agl = calculate_flight_height(
        camera["sensor_width_mm"],
        camera["focal_length_mm"],
        target_gsd,
        camera["image_width_px"]
    )

    # Calculate minimum AGL altitude for GSD = 3.0 cm/pixel
    min_gsd = 3.0
    min_alt_agl = calculate_flight_height(
        camera["sensor_width_mm"],
        camera["focal_length_mm"],
        min_gsd,
        camera["image_width_px"]
    )

    st.write(f"Camera: {selected_camera}, Target GSD: {target_gsd} cm/px, Required AGL Altitude: {required_alt_agl:.2f} m")
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_steps, "Calculated GSD requirements")

    survey_start = lines[0][0]
    survey_end = lines[-1][-1]

    # Pre-calculate safe altitude
    first_seg = lines[0]
    a, b = first_seg[0], first_seg[-1]
    brg_seg = calculate_bearing(a, b)
    entry = move_point_along_bearing(a, (brg_seg + 180) % 360, turning_length)
    temp_prev_wp = move_point_along_bearing(home_pt, calculate_bearing(home_pt, survey_start), 300)
    point3 = (temp_prev_wp[0], temp_prev_wp[1])
    total_dist = geodesic(point3, entry).meters
    num_samples = max(40, int(total_dist / 5))
    brg_to_entry = calculate_bearing(point3, entry)
    sample_coords = [
        move_point_along_bearing(point3, brg_to_entry, total_dist * (i / num_samples))
        for i in range(num_samples + 1)
    ]
    elevs = fetch_elevations(sample_coords)
    valid_elevs = [e for e in elevs if e is not None and -500 < e < 9000]
    if is_unusual_terrain(elevs):
        st.warning("Unusual terrain detected for initial climb to 4th waypoint. Using fallback safety margin.")
        max_elev = home_elev
        max_elev_coords = home_pt
        margin = safety_margin if safety_margin is not None and safety_margin > 0 else FALLBACK_SAFETY_MARGIN
        safe_alt_rel = margin
    else:
        max_elev = max(valid_elevs) if valid_elevs else home_elev
        max_elev_coords = sample_coords[int(np.argmax([e if e is not None and -500 < e < 9000 else -float('inf') for e in elevs]))]
        safe_alt_rel = (max_elev - home_elev) + 150
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_steps, "Calculated safe altitude for initial climb")

    # Take-off ladder
    prev_wp = home_pt
    takeoff_ladder = [(0, home_alt), (300, 90)]
    for dist, alt_rel in takeoff_ladder:
        if dist == 0:
            path.append((prev_wp[0], prev_wp[1], alt_rel))
        else:
            brg = calculate_bearing(prev_wp, survey_start)
            prev_wp = move_point_along_bearing(prev_wp, brg, dist)
            path.append((prev_wp[0], prev_wp[1], alt_rel))
        trigger_points.append(
            {
                "lat": prev_wp[0],
                "lon": prev_wp[1],
                "alt": alt_rel,
                "trigger_type": "none",
                "trigger_params": {},
            }
        )
        item_id_counter += 1
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_steps, "Added takeoff waypoints")

    # Third waypoint (loiter)
    brg = calculate_bearing(prev_wp, survey_start)
    prev_wp = move_point_along_bearing(prev_wp, brg, 500)
    path.append((prev_wp[0], prev_wp[1], safe_alt_rel))
    trigger_points.append(
        {
            "lat": prev_wp[0],
            "lon": prev_wp[1],
            "alt": safe_alt_rel,
            "trigger_type": "loiter",
            "trigger_params": {},
        }
    )
    item_id_counter += 1
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_steps, "Added loiter waypoint")

    # Pre-entry loiter adjustments
    pre_entry_loiter = move_point_along_bearing(entry, (brg_to_entry + 180) % 360, 500)
    bearing_to_pre_entry = calculate_bearing(prev_wp, pre_entry_loiter)
    inverse_bearing = (bearing_to_pre_entry + 180) % 360
    new_waypoint_4 = move_point_along_bearing(pre_entry_loiter, inverse_bearing, 300)
    path.append((new_waypoint_4[0], new_waypoint_4[1], safe_alt_rel))
    trigger_points.append(
        {
            "lat": new_waypoint_4[0],
            "lon": new_waypoint_4[1],
            "alt": safe_alt_rel,
            "trigger_type": "none",
            "trigger_params": {},
        }
    )
    item_id_counter += 1

    # 5th waypoint (pre-entry loiter at cruise altitude)
    first_seg = lines[0]
    a, b = first_seg[0], first_seg[-1]
    brg = calculate_bearing(a, b)
    seg_length = geodesic(a, b).meters
    num_samples = max(10, int(seg_length / 5))
    sample_coords = [move_point_along_bearing(a, brg, seg_length * (i / num_samples)) for i in range(num_samples + 1)]
    first_seg_elevs = fetch_elevations(sample_coords)
    valid_first_seg_elevs = [e for e in first_seg_elevs if e is not None and -500 < e < 9000]
    if not valid_first_seg_elevs:
        st.warning("No valid elevations for first segment at pre-entry loiter. Using fallback safety margin.")
        cruise_rel_first = 150 + (safety_margin if safety_margin is not None and safety_margin > 0 else FALLBACK_SAFETY_MARGIN)
    else:
        max_elev = max(valid_first_seg_elevs)
        min_elev = min(valid_first_seg_elevs)
        cruise_rel_first = max(
            required_alt_agl + max_elev - home_elev,
            min_alt_agl + min_elev - home_elev,
            100
        )
    path.append((pre_entry_loiter[0], pre_entry_loiter[1], cruise_rel_first))
    trigger_points.append(
        {
            "lat": pre_entry_loiter[0],
            "lon": pre_entry_loiter[1],
            "alt": cruise_rel_first,
            "trigger_type": "loiter",
            "trigger_params": {},
        }
    )
    item_id_counter += 1
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_steps, "Added pre-entry loiter waypoint")

    prev_exit, prev_alt_rel = None, cruise_rel_first

    # Survey lines
    for seg_idx, seg in enumerate(lines):
        seg_length = sum(geodesic(seg[i], seg[i + 1]).meters for i in range(len(seg) - 1))
        num_samples = max(30, int(seg_length / 5))
        a, b = seg[0], seg[-1]
        brg = calculate_bearing(a, b)
        sample_coords = [move_point_along_bearing(a, brg, seg_length * (i / num_samples)) for i in range(num_samples + 1)]
        seg_elevs = fetch_elevations(sample_coords)
        valid_seg_elevs = [e for e in seg_elevs if e is not None and -500 < e < 9000]
        if not valid_seg_elevs:
            st.warning(f"No valid elevations for segment {seg_idx + 1} at coords {seg}. Using fallback safety margin.")
            cruise_rel = 150 + (safety_margin if safety_margin is not None and safety_margin > 0 else FALLBACK_SAFETY_MARGIN)
        else:
            max_elev = max(valid_seg_elevs)
            min_elev = min(valid_seg_elevs)
            cruise_rel = max(
                required_alt_agl + max_elev - home_elev,
                min_alt_agl + min_elev - home_elev,
                100
            )
        entry = move_point_along_bearing(a, (brg + 180) % 360, turning_length)
        exitpt = move_point_along_bearing(b, brg, turning_length)
        if prev_exit is not None:
            path.append((prev_exit[0], prev_exit[1], prev_alt_rel))
            trigger_points.append(
                {
                    "lat": prev_exit[0],
                    "lon": prev_exit[1],
                    "alt": prev_alt_rel,
                    "trigger_type": "none",
                    "trigger_params": {},
                }
            )
            item_id_counter += 1
            path.append((entry[0], entry[1], cruise_rel))
            trigger_points.append(
                {
                    "lat": entry[0],
                    "lon": entry[1],
                    "alt": cruise_rel,
                    "trigger_type": "none",
                    "trigger_params": {},
                }
            )
            item_id_counter += 1
        path.extend(
            [
                (entry[0], entry[1], cruise_rel),
                (a[0], a[1], cruise_rel),
                (b[0], b[1], cruise_rel),
                (exitpt[0], exitpt[1], cruise_rel),
            ]
        )
        trigger_points.extend(
            [
                {
                    "lat": entry[0],
                    "lon": entry[1],
                    "alt": cruise_rel,
                    "trigger_type": "none",
                    "trigger_params": {},
                },
                {
                    "lat": a[0],
                    "lon": a[1],
                    "alt": cruise_rel,
                    "trigger_type": "camera",
                    "trigger_params": {"distance": trigger_distance},
                },
                {
                    "lat": b[0],
                    "lon": b[1],
                    "alt": cruise_rel,
                    "trigger_type": "camera",
                    "trigger_params": {"distance": end_trigger_distance},
                },
                {
                    "lat": exitpt[0],
                    "lon": exitpt[1],
                    "alt": cruise_rel,
                    "trigger_type": "none",
                    "trigger_params": {},
                },
            ]
        )
        item_id_counter += 4
        prev_exit, prev_alt_rel = exitpt, cruise_rel
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps, f"Processed survey segment {seg_idx + 1}/{len(lines)}")

    # Calculate last bearing for loiter point
    last_a, last_b = lines[-1][0], lines[-1][-1]
    last_brg = calculate_bearing(last_a, last_b)

    # RTL logic
    exit_pt = (prev_exit[0], prev_exit[1])
    brg_to_home = calculate_bearing(exit_pt, home_pt)
    direct_dist = geodesic(exit_pt, home_pt).meters
    sample_coords = [move_point_along_bearing(exit_pt, brg_to_home, direct_dist * (i / 50)) for i in range(51)]
    elevs = fetch_elevations(sample_coords)
    valid_elevs = [e for e in elevs if e is not None and -500 < e < 9000]
    if is_unusual_terrain(elevs):
        st.warning("Unusual terrain detected for return path. Using fallback safety margin.")
        max_elev = home_elev
        max_elev_coords = home_pt
        new_alt_rel = safety_margin if safety_margin > 0 else FALLBACK_SAFETY_MARGIN
    else:
        max_elev = max(valid_elevs)
        max_elev_coords = sample_coords[int(np.argmax([e if e is not None and -500 < e < 9000 else -float('inf') for e in elevs]))]
        new_alt_rel = (max_elev - home_elev) + 150
    
    # Loiter 300m beyond exit_pt
    loiter_point = move_point_along_bearing(exit_pt, brg_to_home, 300)
    path.append((loiter_point[0], loiter_point[1], new_alt_rel))
    trigger_points.append(
        {
            "lat": loiter_point[0],
            "lon": loiter_point[1],
            "alt": new_alt_rel,
            "trigger_type": "loiter",
            "trigger_params": {},
        }
    )
    item_id_counter += 1
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_steps, "Added RTL loiter waypoint")

    # Add new waypoint and loiter at loiter_180m_pos
    inverse_brg = (brg_to_home + 180) % 360
    loiter_180m_pos = move_point_along_bearing(home_pt, inverse_brg, 800)
    bearing_to_loiter_180m = calculate_bearing(loiter_point, loiter_180m_pos)
    inverse_bearing = (bearing_to_loiter_180m + 180) % 360
    new_waypoint = move_point_along_bearing(loiter_180m_pos, inverse_bearing, 300)
    path.append((new_waypoint[0], new_waypoint[1], new_alt_rel))
    trigger_points.append(
        {
            "lat": new_waypoint[0],
            "lon": new_waypoint[1],
            "alt": new_alt_rel,
            "trigger_type": "none",
            "trigger_params": {},
        }
    )
    item_id_counter += 1
    path.append((loiter_180m_pos[0], loiter_180m_pos[1], 100))
    trigger_points.append(
        {
            "lat": loiter_180m_pos[0],
            "lon": loiter_180m_pos[1],
            "alt": 100,
            "trigger_type": "loiter",
            "trigger_params": {},
        }
    )
    item_id_counter += 1
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_steps, "Added final loiter waypoints")

    # Landing ladder
    landing_ladder = [(300, 90), (300, 0)]
    for dist_from_home, alt_rel in landing_ladder:
        pos = move_point_along_bearing(home_pt, inverse_brg, dist_from_home)
        path.append((pos[0], pos[1], alt_rel))
        trigger_points.append(
            {
                "lat": pos[0],
                "lon": pos[1],
                "alt": alt_rel,
                "trigger_type": "land" if alt_rel == 0 else "none",
                "trigger_params": {},
            }
        )
        item_id_counter += 1
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_steps, "Completed landing waypoints")

    return path, trigger_points, expected_photo_count


# -----------------------------------------------------------------------------
# 📊  GSD & PROFILE HELPERS (these work in MSL space)
# -----------------------------------------------------------------------------

def calculate_mission_stats(path_msl):
    total_distance = sum(
        geodesic(path_msl[i][:2], path_msl[i + 1][:2]).meters for i in range(len(path_msl) - 1)
    )
    cruise_speed = 15  # m/s
    flight_time = total_distance / cruise_speed / 60  # minutes
    coords = [(p[0], p[1]) for p in path_msl]
    terrain_elevs = fetch_elevations(coords)
    altitudes = [p[2] for p in path_msl]
    distances = [0]
    for i in range(1, len(path_msl)):
        d = geodesic(path_msl[i - 1][:2], path_msl[i][:2]).meters
        distances.append(distances[-1] + d)
    return {
        "total_distance": total_distance,
        "flight_time": flight_time,
        "terrain_elevations": terrain_elevs,
        "altitudes": altitudes,
        "distances": distances,
    }

def get_alt_every_20m(path_msl, distances):
    total_distance = distances[-1]
    steps = np.arange(0, total_distance + 1e-6, 10)
    alt_20m = []
    for D in steps:
        i = np.searchsorted(distances, D, side="left")
        if i == len(distances):
            alt_20m.append(path_msl[-1][2])
        elif distances[i] == D:
            alt_20m.append(path_msl[i][2])
        else:
            j = i - 1
            f = (D - distances[j]) / (distances[i] - distances[j])
            alt_D = path_msl[j][2] + f * (path_msl[i][2] - path_msl[j][2])
            alt_20m.append(alt_D)
    return steps, alt_20m

def calculate_mission_gsd(path_msl, trigger_points_msl, distances, camera, terrain_elevations):
    gsd_values = []
    for i in range(len(trigger_points_msl) - 1):
        if (
            trigger_points_msl[i]["trigger_type"] == "camera"
            and trigger_points_msl[i + 1]["trigger_type"] == "camera"
        ):
            start_idx = i
            end_idx = i + 1
            start_dist = distances[start_idx]
            end_dist = distances[end_idx]
            segment_length = end_dist - start_dist
            steps = np.arange(0, segment_length + 1e-6, 10)
            # Generate coordinates for each step
            segment_coords = [get_coord_at_distance(path_msl, distances, start_dist + step) for step in steps]
            # Fetch terrain elevations for these coordinates
            segment_terrain = fetch_elevations(segment_coords)
            for step, terrain in zip(steps, segment_terrain):
                # Use home_elev as fallback for None terrain
                terrain = terrain if terrain is not None and terrain > 0 else home_elev
                # Interpolate altitude (MSL) between start and end points
                f = step / segment_length if segment_length > 0 else 0
                alt_asl = path_msl[start_idx][2] + f * (
                    path_msl[end_idx][2] - path_msl[start_idx][2]
                )
                alt_agl = max(alt_asl - terrain, 0)
                # Calculate GSD in cm/pixel
                gsd = (
                    (alt_agl * camera["sensor_width_mm"]) /
                    (camera["focal_length_mm"] * camera["image_width_px"])
                ) * 100  # Convert to cm/pixel
                gsd_values.append(gsd)
                # Debug
                # if gsd < 3.0:
                #     st.write(f"Warning: GSD {gsd:.2f} cm/px at distance {start_dist + step:.2f} m (AGL: {alt_agl:.2f} m)")
    return np.mean(gsd_values) if gsd_values else np.nan
# -----------------------------------------------------------------------------
# 🗺  QGC JSON EXPORTER (takes RELATIVE altitudes)
# -----------------------------------------------------------------------------

def generate_qgc_plan(points_rel, trigger_points_rel, expected_photo_count, trigger_distance, end_trigger_distance):
    items = []
    item_id = 1

    # Take‑off (command 84)
    items.append(
        {
            "AMSLAltAboveTerrain": None,
            "Altitude": points_rel[0][2],
            "AltitudeMode": 1,
            "autoContinue": True,
            "command": 84,
            "doJumpId": item_id,
            "frame": 3,
            "params": [0, 0, 0, None, 0, 0, points_rel[0][2]],
            "type": "SimpleItem",
        }
    )
    item_id += 1

    for i, (lat, lon, alt_rel) in enumerate(points_rel[1:], start=1):
        tp = trigger_points_rel[i]
        if tp["trigger_type"] == "loiter":
            items.append(
                {
                    "Altitude": alt_rel,
                    "AMSLAltAboveTerrain": None,
                    "AltitudeMode": 1,
                    "autoContinue": True,
                    "command": 31,  # loiter
                    "doJumpId": item_id,
                    "frame": 3,
                    "params": [1, 250, 0, 1, lat, lon, alt_rel],
                    "type": "SimpleItem",
                }
            )
            item_id += 1
        elif tp["trigger_type"] == "land":
            items.append(
                {
                    "Altitude": alt_rel,
                    "AMSLAltAboveTerrain": None,
                    "AltitudeMode": 1,
                    "autoContinue": True,
                    "command": 20,  # RTL
                    "doJumpId": item_id,
                    "frame": 2,  # MAV_FRAME_MISSION
                    "params": [0, 0, 0, 0, 0, 0, 0],
                    # "params": [0, 0, 0, 0, lat, lon, alt_rel],
                    "type": "SimpleItem",
                }
            )
            item_id += 1
        else:
            items.append(
                {
                    "Altitude": alt_rel,
                    "AMSLAltAboveTerrain": None,
                    "AltitudeMode": 1,
                    "autoContinue": True,
                    "command": 16,  # NAV_WAYPOINT
                    "doJumpId": item_id,
                    "frame": 3,
                    "params": [0, 0, 0, None, lat, lon, alt_rel],
                    "type": "SimpleItem",
                }
            )
            item_id += 1
            if tp["trigger_type"] == "camera":
                trig_item = create_trigger_item(
                    tp["lat"], tp["lon"], alt_rel, tp["trigger_type"], tp["trigger_params"].get("distance", trigger_distance), item_id
                )
                if trig_item:
                    items.append(trig_item)
                    item_id += 1

    plan = {
        "fileType": "Plan",
        "version": 1,
        "groundStation": "QGroundControl",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "rallyPoints": {"points": [], "version": 2},
        "mission": {
            "version": 2,
            "firmwareType": 3,
            "vehicleType": 1,
            "cruiseSpeed": 20,
            "hoverSpeed": 5,
            "plannedHomePosition": [points_rel[0][0], points_rel[0][1], 0],
            "items": items,
            "surveyStats": {
                "surveyArea": 0,
                "triggerDistance": trigger_distance,
                "photoCount": expected_photo_count,
            },
        },
    }
    return plan

# -----------------------------------------------------------------------------
# 🎯  FRONT‑END (Streamlit)
# -----------------------------------------------------------------------------

# Streamlit front-end
st.set_page_config(page_title="Drone Flight Plan Generator", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f5f7fa; }
    .sidebar .sidebar-content { background-color: #e9ecef; }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover { background-color: #0056b3; }
    h1 { color: #343a40; }
    h2, h3 { color: #495057; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Drone Flight Plan Generator ✈️")

# Sidebar – user input
with st.sidebar:
    st.header("Input Parameters")
    kml_file = st.file_uploader("Upload KML File", type=["kml"])
    coord_format = st.radio("Coordinate Format", ["Decimal Degrees", "DMS"])
    st.subheader("Home Point")
    if coord_format == "Decimal Degrees":
        hl = st.number_input("Latitude", -90.0, 90.0, value=0.0, format="%.6f")
        hlon = st.number_input("Longitude", -180.0, 180.0, value=0.0, format="%.6f")
    else:
        hl_dms = st.text_input("Latitude DMS (e.g., 18°26'40.72\")")
        hl_dir = st.selectbox("Lat Dir", ["N", "S"], key="hl_dir")
        hlon_dms = st.text_input("Longitude DMS (e.g., 73°52'15.12\")")
        hlon_dir = st.selectbox("Lon Dir", ["E", "W"], key="hlon_dir")

    use_home_for_rtl = st.checkbox("Use Home Point for RTL", value=True)
    if not use_home_for_rtl:
        st.subheader("RTL Point")
        if coord_format == "Decimal Degrees":
            el = st.number_input("RTL Latitude", -90.0, 90.0, value=0.0, format="%.6f")
            elon = st.number_input("RTL Longitude", -180.0, 180.0, value=0.0, format="%.6f")
        else:
            el_dms = st.text_input("RTL Lat DMS", key="el_dms")
            el_dir = st.selectbox("RTL Lat Dir", ["N", "S"], key="el_dir")
            elon_dms = st.text_input("RTL Lon DMS", key="elon_dms")
            elon_dir = st.selectbox("RTL Lon Dir", ["E", "W"], key="elon_dir")

    home_alt = st.number_input("Home Altitude AGL (m)", value=40.0)
    safety_margin = st.number_input("Increase elevation (m)", value=0.0)
    turning_length = st.number_input("Turning Length (m)", value=250.0)
    trigger_distance = st.number_input("Start Trigger Distance (m)", value=40.0)
    end_trigger_distance = st.number_input("End Trigger Distance (m)", value=0.0)
    reverse_kml = st.checkbox("Reverse KML Path", value=False)

    camera_options = {
        "Sony A6000": {
            "sensor_height_mm": 15.6,
            "sensor_width_mm": 23.5,
            "focal_length_mm": 16.0,
            "image_height_px": 4000,
            "image_width_px": 6000,
        },
        "Oblique 3d": {
            "sensor_height_mm": 15.6,
            "sensor_width_mm": 23.5,
            "focal_length_mm": 25.0,
            "image_height_px": 4000,
            "image_width_px": 6000,
        },
        "Sony 42mp": {
            "sensor_height_mm": 23.88,
            "sensor_width_mm": 35.81,
            "focal_length_mm": 35.0,
            "image_height_px": 5304,
            "image_width_px": 7952,
        },
    }
    selected_camera = st.selectbox("Camera for GSD", list(camera_options.keys()))

# Main action button with progress bar
if st.button("Generate Flight Plan"):
    if kml_file is None:
        st.error("No KML file uploaded.")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = 10  # Adjust based on major tasks
    current_step = 0

    def update_progress(progress, message):
        progress_bar.progress(min(progress, 1.0))
        status_text.text(message)
        time.sleep(0.1)  # Smooth animation

    update_progress(current_step / total_steps, "Validating inputs...")
    # Resolve home / RTL coordinates
    if coord_format == "Decimal Degrees":
        home_pt = (hl, hlon)
        rtl_pt = home_pt if use_home_for_rtl else (el, elon)
    else:
        hl_decimal = dms_to_decimal(hl_dms, hl_dir)
        hlon_decimal = dms_to_decimal(hlon_dms, hlon_dir)
        if hl_decimal is None or hlon_decimal is None:
            st.error("Invalid home point coordinates.")
            st.stop()
        home_pt = (hl_decimal, hlon_decimal)
        if use_home_for_rtl:
            rtl_pt = home_pt
        else:
            el_decimal = dms_to_decimal(el_dms, el_dir)
            elon_decimal = dms_to_decimal(elon_dms, elon_dir)
            if el_decimal is None or elon_decimal is None:
                st.error("Invalid RTL point coordinates.")
                st.stop()
            rtl_pt = (el_decimal, elon_decimal)
    current_step += 1
    update_progress(current_step / total_steps, "Resolved coordinates")

    # Home elevation
    home_elev = fetch_elevations([home_pt])[0]
    if home_elev is None or home_elev <= 0:
        st.warning("Invalid home elevation. Using 0 as fallback.")
        home_elev = 0
    current_step += 1
    update_progress(current_step / total_steps, "Fetched home elevation")

    # Parse KML
    kml_file.seek(0)
    coords = parse_kml(kml_file)
    kml_file.seek(0)
    alt_rel = parse_kml_alt_rel(kml_file)
    if not coords:
        st.error("No valid coordinates found in KML file.")
        st.stop()
    if reverse_kml:
        coords.reverse()
    current_step += 1
    update_progress(current_step / total_steps, "Parsed KML file")

    # Process segments
    segs = split_lines_by_turn(coords)
    mains = filter_main_lines(segs)
    if not mains:
        st.error("No valid survey segments found in KML file.")
        st.stop()
    mains = adjust_line_directions(mains)
    current_step += 1
    update_progress(current_step / total_steps, "Processed survey segments")

    # Terrain classification
    terrain_elevs = fetch_elevations(coords)
    elev_diffs = [abs(terrain_elevs[i] - terrain_elevs[i + 1]) for i in range(len(terrain_elevs) - 1) if terrain_elevs[i] is not None and terrain_elevs[i + 1] is not None]
    if elev_diffs:
        max_elev_diff = max(elev_diffs)
        terrain_threshold = 100.00
        if max_elev_diff < terrain_threshold:
            terrain_message = "Flat Terrain (e.g., cities, residential areas)"
            message_color = "green"
        else:
            terrain_message = "Non-Flat Terrain (e.g., hills, mountains) Better to use Increase elevation"
            message_color = "red"
    else:
        terrain_message = "Unable to determine terrain type due to missing elevation data"
        message_color = "orange"
    st.markdown(f"<p style='color: {message_color}; font-family: Segoe UI, sans-serif; font-size: 14px; margin-top: 5px;'>Terrain Type: {terrain_message}</p>", unsafe_allow_html=True)
    current_step += 1
    update_progress(current_step / total_steps, "Classified terrain")

    # Build relative-altitude path
    path_rel, trigger_points_rel, expected_photo_count = generate_simplified_path(
        mains,
        home_pt,
        rtl_pt,
        home_elev,
        home_alt,
        safety_margin,
        turning_length,
        trigger_distance,
        end_trigger_distance,
        kml_alt_rel=alt_rel if alt_rel is not None else 100,
        camera=camera_options[selected_camera],
        progress_callback=lambda p, m: update_progress(current_step / total_steps + (p / total_steps), m)
    )
    current_step += 1
    update_progress(current_step / total_steps, "Generated flight path")

    # Convert to MSL
    path_msl = [(lat, lon, alt_rel + home_elev) for lat, lon, alt_rel in path_rel]
    trigger_points_msl = []
    for tp in trigger_points_rel:
        tp_msl = tp.copy()
        tp_msl["alt"] = tp["alt"] + home_elev
        trigger_points_msl.append(tp_msl)
    current_step += 1
    update_progress(current_step / total_steps, "Converted to MSL altitudes")

    # Mission stats & profiles
    stats = calculate_mission_stats(path_msl)
    current_step += 1
    update_progress(current_step / total_steps, "Calculated mission statistics")

    # Fetch weather data
    weather_data = fetch_weather_data(home_pt[0], home_pt[1])
    current_step += 1
    update_progress(current_step / total_steps, "Fetched weather data")

    # Generate QGC plan
    plan = generate_qgc_plan(
        path_rel, trigger_points_rel, expected_photo_count, trigger_distance, end_trigger_distance
    )
    plan_json = json.dumps(plan, indent=2)
    current_step += 1
    update_progress(1.0, "Flight plan generated successfully!")

    # Store in session state
    st.session_state.path_msl = path_msl
    st.session_state.trigger_points_msl = trigger_points_msl
    st.session_state.stats = stats
    st.session_state.weather_data = weather_data
    st.session_state.plan_json = plan_json
    st.session_state.expected_photo_count = expected_photo_count
    st.session_state.home_pt = home_pt
    st.session_state.rtl_pt = rtl_pt
    st.session_state.use_home_for_rtl = use_home_for_rtl
    st.session_state.home_elev = home_elev
    st.session_state.camera = camera_options[selected_camera]
    st.session_state.plan_generated = True

# Display the columns if the plan has been generated
if st.session_state.plan_generated:
    with st.spinner("Rendering flight plan …"):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🗺 Flight Path Preview")
            m = folium.Map(location=[st.session_state.home_pt[0], st.session_state.home_pt[1]], zoom_start=12, tiles=None)
            folium.TileLayer(
                tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                attr="Map data © Google",
                name="Satellite Streets",
            ).add_to(m)
            folium.Marker(
                [st.session_state.home_pt[0], st.session_state.home_pt[1]],
                popup="Home",
                icon=folium.Icon(color="green"),
            ).add_to(m)
            if not st.session_state.use_home_for_rtl:
                folium.Marker(
                    [st.session_state.rtl_pt[0], st.session_state.rtl_pt[1]],
                    popup="RTL",
                    icon=folium.Icon(color="red"),
                ).add_to(m)
            folium.PolyLine(
                [(lat, lon) for lat, lon, _ in st.session_state.path_msl],
                color="orange",
                weight=2.5
            ).add_to(m)
            folium_static(m, width=700, height=400)

            st.subheader("📈 Mission Profile (MSL)")
            steps, altitudes_msl = get_alt_every_20m(
                st.session_state.path_msl, st.session_state.stats["distances"]
            )
            step_coords = [
                get_coord_at_distance(st.session_state.path_msl, st.session_state.stats["distances"], D)
                for D in steps
            ]
            terrain_elevs = fetch_elevations(step_coords)
            gsd_values = [
                (
                    (max(alt - (terr if terr is not None and terr > 0 else st.session_state.home_elev), 0) * st.session_state.camera["sensor_width_mm"]) /
                    (st.session_state.camera["focal_length_mm"] * st.session_state.camera["image_width_px"])
                ) * 100
                for alt, terr in zip(altitudes_msl, terrain_elevs)
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=steps, y=terrain_elevs, name="Terrain", line=dict(color="#28a745")))
            fig.add_trace(go.Scatter(x=steps, y=altitudes_msl, name="Mission", line=dict(color="#ff9d00")))
            fig.add_trace(
                go.Scatter(
                    x=steps, y=gsd_values, name="GSD (cm/px)", yaxis="y2", line=dict(color="#dc3545")
                )
            )
            fig.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Altitude MSL (m)",
                yaxis2=dict(
                    title="GSD (cm/px)", overlaying="y", side="right", range=[0, max(gsd_values) * 1.2]
                ),
                template="plotly_white",
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("💾 Download Plan")
            st.download_button(
                "Download .plan",
                st.session_state.plan_json,
                file_name="flight_plan.plan",
                mime="application/json"
            )

            st.subheader("🌤 Weather at Home")
            if st.session_state.weather_data:
                cur = st.session_state.weather_data["current"]
                st.write(
                    f"{cur['description'].capitalize()}, {cur['temperature']:.1f}°C"
                )
                st.write(f"- Wind: {cur['wind_speed']:.1f} m/s, {deg_to_cardinal(cur['wind_deg'])}")
                st.write(f"- Humidity: {cur['humidity']}%")
                st.write(f"- Precip (next 3h): {st.session_state.weather_data['short_term_pop']:.0f}%")
                st.write("*3-Day Forecast:*")
                for d in st.session_state.weather_data["forecast"]:
                    st.write(
                        f"- {d['date']}: {d['temp_min']:.1f}°C – {d['temp_max']:.1f}°C, {d['description'].capitalize()}, Precip: {d['pop']:.0f}%"
                    )
            else:
                st.write("Weather data unavailable.")

            st.subheader("📊 Mission Statistics")
            st.write(f"- Total Distance: {st.session_state.stats['total_distance']/1000:.2f} km")
            st.write(f"- Flight Time: {st.session_state.stats['flight_time']:.2f} min")
            mission_gsd = calculate_mission_gsd(
                st.session_state.path_msl,
                st.session_state.trigger_points_msl,
                st.session_state.stats["distances"],
                st.session_state.camera,
                st.session_state.stats["terrain_elevations"],
            )
            st.write(f"- Survey GSD (avg): {mission_gsd:.2f} cm/px")

            loiter_indices = [
                i for i, tp in enumerate(st.session_state.trigger_points_msl)
                if tp["trigger_type"] == "loiter"
            ]
            if len(loiter_indices) >= 2:
                start_idx = loiter_indices[0]
                end_idx = loiter_indices[-1]
                segment_path = st.session_state.path_msl[start_idx:end_idx + 1]
                path_coords = [(lat, lon) for lat, lon, _ in segment_path]
                ground_elevs = fetch_elevations(path_coords)
                agl_heights = [
                    alt_msl - ground_elev for (lat, lon, alt_msl), ground_elev
                    in zip(segment_path, ground_elevs) if ground_elev is not None
                ]
                if agl_heights:
                    st.write(f"- Max AGL Altitude: {max(agl_heights):.2f} m")
                    st.write(f"- Min AGL Altitude: {min(agl_heights):.2f} m")
                    st.write(f"- Avg AGL Altitude: {sum(agl_heights) / len(agl_heights):.2f} m")
                else:
                    st.warning("Unable to calculate mission elevations due to missing data.")
            else:
                st.warning("Insufficient loiter waypoints for elevation stats.")

            # st.subheader("💾 Download Plan")
            # st.download_button(
            #     "Download .plan",
            #     st.session_state.plan_json,
            #     file_name="flight_plan.plan",
            #     mime="application/json"
            # )


