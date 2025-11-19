"""Utility functions for the Luna integration."""

from __future__ import annotations

import datetime
import logging
import math
from datetime import timezone

import ephem
from homeassistant.core import HomeAssistant
from homeassistant.util import slugify

from .const import DOMAIN, DEBUG_ELEVATION_SENSOR, ELEVATION_TOLERANCE, AZIMUTH_DEGREE_TOLERANCE, AZIMUTH_REVERSAL_SEARCH_MAX_ITERATIONS, AZIMUTH_TERNARY_SEARCH_MAX_ITERATIONS
from .config_store import get_config_entry_data

_LOGGER = logging.getLogger(__name__)


def get_moon_phase(phase_percent: float | None) -> tuple[str, str]:
    """
    Get the moon phase name and icon based on phase_percent.
    
    Args:
        phase_percent: Position in lunar cycle (0-100, where 0=previous new moon, 100=next new moon)
    
    Returns:
        Tuple of (phase_name, icon_name):
        - phase_name: "New Moon", "Waxing Crescent", "First Quarter", etc.
        - icon_name: "mdi:moon-new", "mdi:moon-waxing-crescent", "mdi:moon-first-quarter", etc.
    """
    if phase_percent is None:
        return ("New Moon", "mdi:moon-new")
    
    rounded_phase = math.floor(phase_percent)
    
    if rounded_phase <= 1 or rounded_phase >= 98:
        return ("New Moon", "mdi:moon-new")
    elif rounded_phase < 23:
        return ("Waxing Crescent", "mdi:moon-waxing-crescent")
    elif rounded_phase >= 23 and rounded_phase <= 26:
        return ("First Quarter", "mdi:moon-first-quarter")
    elif rounded_phase < 48:
        return ("Waxing Gibbous", "mdi:moon-waxing-gibbous")
    elif rounded_phase >= 48 and rounded_phase <= 51:
        return ("Full Moon", "mdi:moon-full")
    elif rounded_phase < 73:
        return ("Waning Gibbous", "mdi:moon-waning-gibbous")
    elif rounded_phase >= 73 and rounded_phase <= 76:
        return ("Last Quarter", "mdi:moon-last-quarter")
    else:
        return ("Waning Crescent", "mdi:moon-waning-crescent")


def normalize_between_datetimes(
    current_dt: datetime.datetime | None,
    start_dt: datetime.datetime | None,
    end_dt: datetime.datetime | None,
    scale: float,
) -> float | None:
    """
    Normalize where current_dt falls between start_dt and end_dt to a 0-scale range.

    Args:
        current_dt: Current datetime.
        start_dt: Start datetime (mapped to 0).
        end_dt: End datetime (mapped to scale).
        scale: The scale to normalize to (e.g., 100 for percent, 360 for degrees).

    Returns:
        Float between 0 and scale (inclusive) or None if normalization can't be computed.
    """

    def _to_utc(dt: datetime.datetime | None) -> datetime.datetime | None:
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    current = _to_utc(current_dt)
    start = _to_utc(start_dt)
    end = _to_utc(end_dt)

    if current is None or start is None or end is None:
        return None

    start_ts = start.timestamp()
    end_ts = end.timestamp()
    current_ts = current.timestamp()

    if end_ts <= start_ts:
        return None

    progress = (current_ts - start_ts) / (end_ts - start_ts)
    progress = max(0.0, min(1.0, progress))

    return progress * scale


def format_sensor_naming(sensor_name: str, entry_id: str) -> tuple[str, str]:
    """
    Create consistent sensor name and unique ID for lunar sensors.
    
    Args:
        sensor_name: The specific name for this sensor (e.g., "lunar elevation", "azimuth angle")
        entry_id: The config entry ID to generate a unique identifier
    
    Returns:
        Tuple of (formatted_sensor_name, unique_id)
        
    Example:
        sensor_name, unique_id = format_sensor_naming("lunar elevation", "entry_123")
        # Returns: ("Lunar Elevation", "luna_lunar_elevation_entry_123") or ("Tokyo - Lunar Elevation", "luna_tokyo_lunar_elevation_entry_123")
    """
    
    
    # Get config data to check for location name
    config_data = get_config_entry_data(entry_id)
    
    
    # Just use the sensor name directly
    name = sensor_name
    location_name = config_data.get("location_name") if config_data else None
    
    
    if location_name is not None:
        formatted_name = f"{location_name.title()} - {sensor_name.title()}"
        # Create unique ID with location prefix
        unique_id = f"{slugify(DOMAIN)}_{slugify(location_name)}_{slugify(sensor_name, separator='_')}_{entry_id}"
        
    else:
        formatted_name = f"{name.title()}"
        # Create unique ID without location prefix for unnamed sensors
        unique_id = f"{slugify(DOMAIN)}_{slugify(sensor_name, separator='_')}_{entry_id}"
        

    return formatted_name, unique_id


def format_input_entity_naming(sensor_name: str, config_variable: str) -> tuple[str, str]:
    """
    Create consistent input entity name and unique ID for configuration variables.
    
    Args:
        sensor_name: The specific name for this sensor (e.g., "lunar elevation", "azimuth angle")
        config_variable: The configuration variable name (e.g., "panel angle", "efficiency")
    
    Returns:
        Tuple of (formatted_input_entity_name, unique_id)
        
    Example:
        input_name, input_id = format_input_entity_naming("lunar elevation", "panel angle")
        # Returns: ("Luna - Lunar Elevation - Panel Angle", "luna_lunar_elevation_panel_angle")
    """
    # Format input entity name: "[DOMAIN.title()] - [sensor_name.title()] - [config_variable.title()]"
    formatted_name = f"{DOMAIN.title()} - {sensor_name.title()} - {config_variable.title()}"
    formatted_unique_id = f"{slugify(DOMAIN)}_{slugify(sensor_name, separator='_')}_{slugify(config_variable, separator='_')}"
    
    return formatted_name, formatted_unique_id


def get_moon_position(
    hass: HomeAssistant,
    dt: datetime.datetime,
    entry_id: str,
    use_center: bool = True,
    config_data: dict = None
) -> dict:
    """
    Calculate current moon position using ephem.
    
    Args:
        hass: Home Assistant instance to get location settings
        dt: Local datetime to calculate position for
        use_center: Whether to use the center of the moon (True) or the edge (False)
    
    Returns:
        Dictionary with moon position data:
        - azimuth: Moon azimuth in degrees (0=North, 90=East, 180=South, 270=West)
        - elevation: Moon elevation in degrees
        - declination: Moon declination in degrees
        - size: Angular size of the moon's disk (in degrees)
        - latitude: Latitude of the location
        - longitude: Longitude of the location
        - elevation_m: Elevation in meters
        - pressure_mbar: Pressure in millibars
        - percent_illuminated: Percent of moon surface illuminated (0-100)
        - phase_percent: Position in lunar cycle (0-100, where 0=previous new moon, 100=next new moon)
        - previous_new_moon: Datetime of previous new moon
        - next_new_moon: Datetime of next new moon
    """
    # Get location from config entry data, fallback to Home Assistant configuration
    if config_data is None:
        config_data = get_config_entry_data(entry_id)
    latitude = config_data.get("latitude", hass.config.latitude) if config_data else hass.config.latitude
    longitude = config_data.get("longitude", hass.config.longitude) if config_data else hass.config.longitude
    elevation = config_data.get("elevation", hass.config.elevation) if config_data else hass.config.elevation
    pressure_mbar = config_data.get("pressure_mbar", 1013.25) if config_data else 1013.25
    
    # Convert local datetime to UTC for ephem
    dt_utc = dt.astimezone(timezone.utc)
    
    # Create observer
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.elevation = elevation
    observer.pressure = pressure_mbar
    observer.date = dt_utc
    
    # Set horizon based on use_center parameter
    if not use_center:
        # Set horizon to 0 for edge of moon calculations
        observer.horizon = '0'
    
    # Create moon object
    moon = ephem.Moon()
    
    # Calculate current position
    moon.compute(observer)
    
    # Convert angles from radians to degrees
    azimuth_deg = math.degrees(moon.az) % 360
    elevation_deg = math.degrees(moon.alt)

    declination_deg = math.degrees(moon.dec)
    


    size_deg = moon.size/3600
    
    # Get percent illuminated (ephem returns 0-1, convert to 0-100)
    percent_illuminated = moon.moon_phase * 100
    
    # Calculate lunar phase percent/degrees relative to previous & next new moon
    prev_new_moon_dt = None
    next_new_moon_dt = None
    phase_percent = None
    phase_degrees = None
    try:
        previous_new_moon = ephem.previous_new_moon(dt_utc)
        next_new_moon = ephem.next_new_moon(dt_utc)

        prev_new_moon_dt = ephem.Date(previous_new_moon).datetime().replace(tzinfo=timezone.utc)
        next_new_moon_dt = ephem.Date(next_new_moon).datetime().replace(tzinfo=timezone.utc)

        phase_percent = normalize_between_datetimes(dt_utc, prev_new_moon_dt, next_new_moon_dt, 100.0)
        phase_degrees = normalize_between_datetimes(dt_utc, prev_new_moon_dt, next_new_moon_dt, 360.0)
    except Exception:
        prev_new_moon_dt = None
        next_new_moon_dt = None
        phase_percent = None
        phase_degrees = None
    
    # Calculate lunar transit (highest point) for today
    try:
        lunar_transit = observer.next_transit(moon)
        lunar_transit_dt = lunar_transit.datetime().replace(tzinfo=timezone.utc)
    except Exception:
        lunar_transit_dt = None

    # Calculate lunar antitransit (lowest point) for today
    try:
        lunar_antitransit = observer.next_antitransit(moon)
        lunar_antitransit_dt = lunar_antitransit.datetime().replace(tzinfo=timezone.utc)
    except Exception:
        lunar_antitransit_dt = None
    
    return {
        "azimuth": azimuth_deg,
        "elevation": elevation_deg,
        "declination": declination_deg,
        "size": size_deg,
        "latitude": latitude,
        "longitude": longitude,
        "elevation_m": elevation,
        "pressure_mbar": pressure_mbar,
        "lunar_transit": lunar_transit_dt,
        "lunar_antitransit": lunar_antitransit_dt,
        "percent_illuminated": round(percent_illuminated, 2),
        "phase_percent": round(phase_percent, 2) if phase_percent is not None else None,
        "phase_degrees": round(phase_degrees, 2) if phase_degrees is not None else None,
        "previous_new_moon": prev_new_moon_dt,
        "next_new_moon": next_new_moon_dt
    }

def get_next_step(
    target_type: str,
    step_value: float,
    moon_data: dict,
    dt: datetime.datetime,
    entry_id: str,
    hass: HomeAssistant,
    debug_flag: bool = False,
    config_data: dict = None
) -> dict | float | None:
    """
    Calculate the next step value for elevation and azimuth updates.
    For azimuth, returns a dict with azimuth, reversal flag, and reversal time.
    For elevation, returns the next step as a float.

    Args:
        target_type: 'elevation' or 'azimuth'
        step_value: The step increment value.
        moon_data: Dictionary containing moon position data with lunar transit and antitransit
        dt: Local datetime to calculate from
        entry_id: Config entry ID for location data
        hass: Home Assistant instance
        debug_flag: Whether to enable debug logging for this call

    Returns:
        For azimuth: dict with keys 'azimuth', 'reversal', and optionally 'reversal_time'.
        For elevation: float value for the next step.
        None if target_type is invalid or current_position is missing.
    """
    dt_utc = dt.astimezone(timezone.utc)


    if target_type == 'elevation':
        current_position = moon_data.get('elevation')
        if current_position is None:
            return None
            
        # Use lunar events cache for more reliable direction determination
        try:
            from .cache import get_cached_lunar_event
            cached_next_event_time, cached_next_event_type, cached_next_event_elevation, used_cache = get_cached_lunar_event(
                hass, entry_id, dt
            )
            
            # Check if we have valid cached data and it's not in the past
            # Convert to timestamps for precise comparison to avoid floating-point precision issues
            if (cached_next_event_time and 
                cached_next_event_type and 
                cached_next_event_elevation is not None):
                
                cached_timestamp = cached_next_event_time.timestamp()
                current_timestamp = dt_utc.timestamp()
                
                if cached_timestamp >= current_timestamp:
                    # Use cached lunar event for direction determination
                    time_difference = cached_timestamp - current_timestamp
                    
                    if cached_next_event_type == 'transit':
                        try:
                            next_step_value = round((round(current_position / step_value) * step_value) + step_value, 2)
                            if next_step_value > cached_next_event_elevation:
                                # Overshoot → pass event elevation and its time to avoid tangent search
                                return {
                                    'elevation': cached_next_event_elevation,
                                    'next_time': cached_next_event_time,
                                    'event': 'transit'
                                }
                            else:
                                # Heading toward lunar transit - elevation increasing
                                if time_difference >= ELEVATION_TOLERANCE:
                                    next_rising_dt, next_setting_dt, next_event_dt = get_time_at_elevation(
                                        hass=hass,
                                        target_elevation=next_step_value,
                                        dt=dt_utc - datetime.timedelta(minutes=360),
                                        entry_id=entry_id,
                                        next_transit_fallback=True,
                                        config_data=config_data,
                                    )
                                    event_label = 'rising' if next_event_dt == next_rising_dt else ('setting' if next_event_dt == next_setting_dt else None)
                                    return {
                                        'elevation': float(next_step_value),
                                        'next_time': next_rising_dt,
                                        'event': 'rising'
                                    }
                                else:
                                    # At or very close to lunar transit - use event elevation and time
                                    return {
                                        'elevation': cached_next_event_elevation,
                                        'next_time': cached_next_event_time,
                                        'event': 'transit'
                                    }
                        except Exception as e:
                            _LOGGER.error(f"Error in lunar transit calculation: current_position={current_position}, step_value={step_value}, cached_next_event_elevation={cached_next_event_elevation}, time_difference={time_difference}, error={e}")
                            # Fallback to cached event elevation
                            return {
                                'elevation': 'Unknown',
                                'next_time': dt_utc + datetime.timedelta(minutes=1),
                                'event': 'Fallback Method'
                            }

                    elif cached_next_event_type == 'antitransit':
                        try:
                            # Heading toward lunar antitransit - elevation decreasing
                            next_step_value = round((round(current_position / step_value) * step_value) - step_value, 2)
                            if next_step_value < cached_next_event_elevation:
                                # Overshoot → pass event elevation and its time to avoid tangent search
                                return {
                                    'elevation': cached_next_event_elevation,
                                    'next_time': cached_next_event_time,
                                    'event': 'antitransit'
                                }
                            else:
                                if time_difference >= ELEVATION_TOLERANCE:
                                    next_rising_dt, next_setting_dt, next_event_dt = get_time_at_elevation(
                                        hass=hass,
                                        target_elevation=next_step_value,
                                        dt=dt_utc - datetime.timedelta(minutes=360),
                                        entry_id=entry_id,
                                        next_transit_fallback=True,
                                        config_data=config_data,
                                    )
                                    event_label = 'rising' if next_event_dt == next_rising_dt else ('setting' if next_event_dt == next_setting_dt else None)
                                    return {
                                        'elevation': float(next_step_value),
                                        'next_time': next_setting_dt,
                                        'event': 'setting'
                                    }
                                else:
                                    # At or very close to lunar antitransit - use event elevation and time
                                    return {
                                        'elevation': cached_next_event_elevation,
                                        'next_time': cached_next_event_time,
                                        'event': 'antitransit'
                                    }
                        except Exception as e:
                            _LOGGER.error(f"Error in lunar antitransit calculation: current_position={current_position}, step_value={step_value}, cached_next_event_elevation={cached_next_event_elevation}, time_difference={time_difference}, error={e}")
                            # Fallback to cached event elevation
                            return {
                                'elevation': 'Unknown',
                                'next_time': dt_utc + datetime.timedelta(minutes=1),
                                'event': 'Fallback Method'
                            }
                    else:
                        # Fallback to current position
                        return {
                            'elevation': 'Unknown',
                            'next_time': dt_utc + datetime.timedelta(minutes=1),
                            'event': 'Fallback Method'
                        }
            else:
                # Cache invalid, stale, or not available - trigger fallback
                raise Exception("Cache invalid or stale")
                    
        except Exception as e:
            # Single fallback logic for all cases:
            # - Cached event is in the past
            # - Cache not available
            # - Any other cache failure
            _LOGGER.debug(f"Using fallback logic for elevation step calculation: {e}")
            return {
                'elevation': 'Unknown',
                'next_time': dt_utc + datetime.timedelta(minutes=1),
                'event': 'Fallback Method'
            }


    elif target_type == 'azimuth':
        current_position = moon_data.get('azimuth')
        if current_position is None:
            return None
        
        # Initialize blocks variable to prevent scope issues
        blocks = False
        
        # Get reversal data for direction detection and reversal checking
        try:
            reversal_data = get_azimuth_reversals(hass, dt.date(), entry_id, config_data=config_data)
            reversals = reversal_data.get('reversals', [])
            base_direction = reversal_data.get('base_direction', 'positive')
        except Exception as e:
            # Fallback to original latitude-based logic
            if config_data is None:
                config_data = get_config_entry_data(entry_id)
            latitude = config_data.get("latitude", hass.config.latitude) if config_data else hass.config.latitude
            declination = moon_data.get('declination')
            if declination is None:
                return None
            base_direction = 1 if latitude > declination else -1
            reversals = []
        
        # Determine current azimuth direction using reversal cache
        try:
            current_direction = get_current_azimuth_direction(dt_utc, base_direction, reversals)
        except Exception as e:
            return None
        
        # Calculate next azimuth step with signed arithmetic
        signed_step = step_value * current_direction
        target_azimuth = (round(current_position / step_value) * step_value + signed_step) % 360
        
        # Check if any reversal occurs before the calculated target
        # If so, use the reversal azimuth instead of the step target
        for i, reversal in enumerate(reversals):
            reversal_time = reversal['time']
            reversal_azimuth = reversal['azimuth']
            
            # Check if we're very close to a reversal point (within 2 degrees)
            azimuth_distance = abs(current_position - reversal_azimuth)
            if azimuth_distance > 180:  # Handle wraparound
                azimuth_distance = 360 - azimuth_distance
            
            is_close_to_reversal = azimuth_distance <= 2.0  # Within 2 degrees
            
            # Consider future reversals OR reversals we're very close to
            if reversal_time > dt_utc or is_close_to_reversal:
                if is_close_to_reversal:
                    # Check if reversal blocks the step target
                    if current_direction == 1:  # Moving positive
                        # Moving positive: check if reversal is between current and target
                        if current_position <= target_azimuth:
                            # No wraparound
                            blocks = current_position < reversal_azimuth < target_azimuth
                        else:
                            # Wraparound case (e.g., 350° -> 10°)
                            blocks = (reversal_azimuth > current_position) or (reversal_azimuth < target_azimuth)
                    else:  # Moving negative (current_direction == -1)
                        # Moving negative: check if reversal is between target and current
                        if target_azimuth <= current_position:
                            # No wraparound
                            blocks = target_azimuth < reversal_azimuth < current_position
                        else:
                            # Wraparound case (e.g., 10° -> 350°)
                            blocks = (reversal_azimuth < current_position) or (reversal_azimuth > target_azimuth)
                
                if blocks:
                    return {
                        "azimuth": reversal_azimuth,
                        "reversal": True,
                        "reversal_time": reversal_time
                    }

                
                break  # Only check the next reversal
            else:
                continue
        
        return {
            "azimuth": target_azimuth,
            "reversal": False
        }

    else:
        return None

def get_time_at_elevation(
    hass: HomeAssistant,
    target_elevation: float,
    dt: datetime.datetime,
    entry_id: str,
    use_center: bool = True,
    next_transit_fallback: bool = False,
    config_data: dict = None
) -> tuple[datetime.datetime | None, datetime.datetime | None, datetime.datetime | None]:
    """
    Calculate when the moon will be at a specific elevation.
    
    Args:
        hass: Home Assistant instance to get location settings
        target_elevation: Target elevation in degrees
        dt: Local datetime to calculate from
        entry_id: Config entry ID for location data
        use_center: Whether to use center of moon (True) or edge of moon (False)
        next_transit_fallback: If True, use transit/antitransit fallback; if False, search 365 days
    
    Returns:
        Tuple of (next_rising, next_setting, next_event)
    """
    # Get location from config entry data
    if config_data is None:
        config_data = get_config_entry_data(entry_id)
    latitude = config_data.get("latitude", hass.config.latitude) if config_data else hass.config.latitude
    longitude = config_data.get("longitude", hass.config.longitude) if config_data else hass.config.longitude
    elevation = config_data.get("elevation", hass.config.elevation) if config_data else hass.config.elevation
    pressure_mbar = config_data.get("pressure_mbar", 1013.25) if config_data else 1013.25
    
    # Convert local datetime to UTC for ephem
    dt_utc = dt.astimezone(timezone.utc)
    
    # Create observer with all settings
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.elevation = elevation
    observer.pressure = pressure_mbar
    observer.date = dt_utc
    observer.horizon = str(target_elevation)  # Set horizon to target elevation
    
    # Create moon object
    moon = ephem.Moon()
    
    try:
        # Try to get rising/setting at target elevation
        next_rising = observer.next_rising(moon, use_center=use_center)
        next_setting = observer.next_setting(moon, use_center=use_center)
        
        # Convert to datetime objects
        next_rising_dt = next_rising.datetime().replace(tzinfo=timezone.utc) if next_rising else None
        next_setting_dt = next_setting.datetime().replace(tzinfo=timezone.utc) if next_setting else None
        
        # Calculate which is next (rising or setting)
        events = [next_rising_dt, next_setting_dt]
        valid_events = [event for event in events if event is not None]
        next_event_dt = min(valid_events) if valid_events else None
        
        return next_rising_dt, next_setting_dt, next_event_dt
        
    except Exception:
        # Target elevation not reachable
        if next_transit_fallback:
            # Get current moon elevation and set as horizon
            current_moon_data = get_moon_position(hass, dt, entry_id, config_data=config_data)
            current_elevation = current_moon_data['elevation']
            observer.horizon = str(current_elevation)
            
            try:
                next_transit = observer.next_transit(moon)
                next_antitransit = observer.next_antitransit(moon)
                
                # Convert to datetime objects
                next_transit_dt = next_transit.datetime().replace(tzinfo=timezone.utc) if next_transit else None
                next_antitransit_dt = next_antitransit.datetime().replace(tzinfo=timezone.utc) if next_antitransit else None
                
                # Find the sooner of transit or antitransit
                events = [next_transit_dt, next_antitransit_dt]
                valid_events = [event for event in events if event is not None]
                next_transit_event = min(valid_events) if valid_events else None
                
                return None, None, next_transit_event
                
            except Exception:
                return None, None, None
        
        else:  # next_transit_fallback == False
            try:
                # Iterate through next 365 days
                search_date = dt_utc
                
                for day in range(365):
                    search_date += datetime.timedelta(days=1)
                    observer.date = search_date
                    
                    try:
                        test_rising = observer.next_rising(moon, use_center=use_center)
                        test_setting = observer.next_setting(moon, use_center=use_center)
                        
                        if test_rising or test_setting:
                            # Found valid rising/setting times
                            next_rising_dt = test_rising.datetime().replace(tzinfo=timezone.utc) if test_rising else None
                            next_setting_dt = test_setting.datetime().replace(tzinfo=timezone.utc) if test_setting else None
                            
                            # Calculate the sooner of rising or setting
                            events = [next_rising_dt, next_setting_dt]
                            valid_events = [event for event in events if event is not None]
                            next_event_dt = min(valid_events) if valid_events else None
                            
                            return next_rising_dt, next_setting_dt, next_event_dt
                    except Exception:
                        continue
                
                # Nothing found in 365 days
                return None, None, None
                
            except Exception:
                return None, None, None


def get_moonrise_moonset_for_date(
    hass: HomeAssistant,
    date: datetime.date,
    entry_id: str,
    use_center: bool = False,
    config_data: dict = None
) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """
    Get moonrise and moonset times for a specific date.
    
    Args:
        hass: Home Assistant instance
        date: Date to calculate for (local date)
        entry_id: Config entry ID for location data
        use_center: Whether to use center (True) or edge (False) of moon
        config_data: Optional config data dict
    
    Returns:
        Tuple of (moonrise_time, moonset_time) in UTC, or (None, None) if not available
    """
    import zoneinfo
    
    # Get location from config entry data
    if config_data is None:
        config_data = get_config_entry_data(entry_id)
    latitude = config_data.get("latitude", hass.config.latitude) if config_data else hass.config.latitude
    longitude = config_data.get("longitude", hass.config.longitude) if config_data else hass.config.longitude
    elevation = config_data.get("elevation", hass.config.elevation) if config_data else hass.config.elevation
    pressure_mbar = config_data.get("pressure_mbar", 1013.25) if config_data else 1013.25
    
    # Get local timezone
    local_tz = zoneinfo.ZoneInfo(hass.config.time_zone)
    
    # Create local midnight for the given date, then convert to UTC
    local_midnight = datetime.datetime.combine(date, datetime.time(0, 0, 0), tzinfo=local_tz)
    midnight_utc = local_midnight.astimezone(timezone.utc)
    
    # Create observer with all settings
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.elevation = elevation
    observer.pressure = pressure_mbar
    observer.date = midnight_utc
    observer.horizon = '0'  # Horizon at 0 degrees
    
    # Create moon object
    moon = ephem.Moon()
    
    try:
        # Get rising and setting for the date (starting from midnight)
        # Use previous_rising/previous_setting to get events that occur on the date
        # We need to check both previous and next to ensure we get events on the target date
        
        # First, try to get next rising/setting from midnight
        try:
            next_rising = observer.next_rising(moon, use_center=use_center)
            next_setting = observer.next_setting(moon, use_center=use_center)
            
            # Convert to datetime objects
            next_rising_dt = next_rising.datetime().replace(tzinfo=timezone.utc) if next_rising else None
            next_setting_dt = next_setting.datetime().replace(tzinfo=timezone.utc) if next_setting else None
            
            # Check if these events are on the target date (within 24 hours from midnight)
            moonrise_time = None
            moonset_time = None
            
            if next_rising_dt:
                next_rising_local = next_rising_dt.astimezone(local_tz)
                if next_rising_local.date() == date:
                    moonrise_time = next_rising_dt
            
            if next_setting_dt:
                next_setting_local = next_setting_dt.astimezone(local_tz)
                if next_setting_local.date() == date:
                    moonset_time = next_setting_dt
            
            # Also check previous rising/setting in case they occur on the target date
            # (e.g., if we're checking at midnight and the event happened just before)
            observer.date = midnight_utc - datetime.timedelta(seconds=1)
            try:
                previous_rising = observer.previous_rising(moon, use_center=use_center)
                previous_setting = observer.previous_setting(moon, use_center=use_center)
                
                if previous_rising:
                    prev_rising_dt = previous_rising.datetime().replace(tzinfo=timezone.utc)
                    prev_rising_local = prev_rising_dt.astimezone(local_tz)
                    if prev_rising_local.date() == date and (moonrise_time is None or prev_rising_dt > moonrise_time):
                        moonrise_time = prev_rising_dt
                
                if previous_setting:
                    prev_setting_dt = previous_setting.datetime().replace(tzinfo=timezone.utc)
                    prev_setting_local = prev_setting_dt.astimezone(local_tz)
                    if prev_setting_local.date() == date and (moonset_time is None or prev_setting_dt > moonset_time):
                        moonset_time = prev_setting_dt
            except Exception:
                pass
            
            return moonrise_time, moonset_time
            
        except Exception:
            return None, None
            
    except Exception:
        return None, None


def get_time_at_azimuth(
    hass: HomeAssistant,
    target_azimuth: float,
    current_dt: datetime.datetime,
    entry_id: str,
    start_dt: datetime.datetime | None = None,
    search_window_hours: float = 6.0,
    config_data: dict = None
) -> tuple[datetime.datetime | None, dict]:
    """
    Find the time when the moon will be at a specific azimuth using ternary search.
    
    Args:
        hass: Home Assistant instance
        target_azimuth: Target azimuth in degrees (0-360)
        current_dt: Current datetime (search will never go before this)
        entry_id: Config entry ID for location data
        start_dt: Starting datetime for search (defaults to nearest lunar transit/antitransit)
        search_window_hours: Initial search window in hours (defaults to 6.0)
    
    Returns:
        Tuple of (datetime when moon will be at target azimuth or None, performance metrics dict)
    """
    # Start timing and iteration tracking
    import time
    start_time = time.time()
    total_iterations = 0
    
    # IMMEDIATELY convert ALL datetime objects to UTC first
    current_dt_utc = current_dt.astimezone(timezone.utc)
    start_dt_utc = start_dt.astimezone(timezone.utc) if start_dt is not None else None
    
    # Get cached reversal data to determine maximum search window
    try:
        current_date = current_dt.date()
        reversal_data = get_azimuth_reversals(hass, current_date, entry_id, config_data=config_data)
        reversals = reversal_data.get('reversals', [])
        
        # Find next future reversal
        next_reversal_time = None
        for reversal in sorted(reversals, key=lambda r: r['time']):
            if reversal['time'] > current_dt_utc:
                next_reversal_time = reversal['time']
                break
        
        # Calculate max search window (cap at next reversal or 24 hours)
        if next_reversal_time:
            seconds_until_reversal = (next_reversal_time - current_dt_utc).total_seconds()
            max_search_seconds = min(24 * 3600, seconds_until_reversal)
        else:
            max_search_seconds = 24 * 3600  # No reversals, use 24h limit
            
    except Exception as e:
        max_search_seconds = 24 * 3600
    
    # Normalize target azimuth to 0-360 range
    target_azimuth = target_azimuth % 360
    
    def normalize_azimuth(azimuth: float) -> float:
        """Convert ephem's 360° to 0° and normalize to 0-360 range."""
        if abs(azimuth - 360.0) < 0.001:
            azimuth = 0.0
        return azimuth % 360
    
    # If no start_dt provided, start search from current time
    if start_dt_utc is None:
        start_dt_utc = current_dt_utc

    # Ensure start_dt is not before current_dt (both in UTC)
    if start_dt_utc < current_dt_utc:
        start_dt_utc = current_dt_utc

    def ternary_search(start_time: datetime.datetime, initial_window_seconds: float) -> tuple[datetime.datetime | None, int]:
        """Perform ternary search for target azimuth."""
        
        # Initialize iteration counter
        iterations = 0
        
        # Initialize window - forward-looking from start_time
        # This ensures we search from current time forward, but can look back if needed during narrowing
        left = start_time
        right = start_time + datetime.timedelta(seconds=initial_window_seconds)
        
        # No need to shift - window is already positioned correctly starting from current time
        
        # Initialize best result tracking
        best_time = None
        best_azimuth_diff = float('inf')
        
        # Direction-agnostic window boundary check
        # Instead of assuming azimuth direction, sample multiple points in the window
        # to see if the target azimuth is reachable
        def is_target_in_window(left_time: datetime.datetime, right_time: datetime.datetime) -> bool:
            """Check if target azimuth is reachable within the time window."""
            
            # Sample more points for better wraparound detection
            sample_times = []
            window_duration = (right_time - left_time).total_seconds()
            num_samples = 10  # Increase from 5 to 10 for better detection
            
            for i in range(num_samples):
                sample_time = left_time + datetime.timedelta(seconds=window_duration * i / (num_samples - 1))
                sample_times.append(sample_time)
            
            # Get azimuth values at sample points
            sample_azimuths = []
            for sample_time in sample_times:
                sample_data = get_moon_position(hass, sample_time, entry_id, config_data=config_data)
                sample_azimuth = normalize_azimuth(sample_data.get('azimuth', 0))
                sample_azimuths.append(sample_azimuth)
            
            # Helper function to check if target is between two azimuths
            def is_target_between_azimuths(az1: float, az2: float, target: float) -> bool:
                """Check if target azimuth is between two azimuth values, handling wraparound."""
                
                # Calculate the azimuth difference
                diff = az2 - az1
                
                # Handle wraparound
                if diff > 180:
                    diff -= 360  # e.g., 350° → 10° becomes -340° → 20°
                elif diff < -180:
                    diff += 360  # e.g., 10° → 350° becomes 340° → -20°
                
                if diff >= 0:  # Moving in positive direction
                    if az1 <= az2:  # No wraparound
                        return az1 <= target <= az2
                    else:  # Wraparound case (e.g., 350° → 10°)
                        return target >= az1 or target <= az2
                else:  # Moving in negative direction
                    if az1 >= az2:  # No wraparound
                        return az2 <= target <= az1
                    else:  # Wraparound case (e.g., 10° → 350°)
                        return target <= az1 or target >= az2
            
            # Check if target is reachable by examining azimuth continuity
            for i in range(len(sample_azimuths) - 1):
                az1 = sample_azimuths[i]
                az2 = sample_azimuths[i + 1]
                
                # Check if target is between consecutive samples
                if is_target_between_azimuths(az1, az2, target_azimuth):
                    return True
            
            return False
        
        # Check if target is within window
        if not is_target_in_window(left, right):
            # Try expanding window if target not found
            expanded_window_seconds = initial_window_seconds * 2
            if expanded_window_seconds <= max_search_seconds:  # Use max_search_seconds
                result, sub_iterations = ternary_search(start_time, expanded_window_seconds)
                return result, iterations + sub_iterations
            return None, iterations

        # Perform ternary search with direction-agnostic narrowing
        while iterations < AZIMUTH_TERNARY_SEARCH_MAX_ITERATIONS:
            iterations += 1
            
            # Calculate test points at 1/3 and 2/3 of window
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            
            # Get azimuths at test points
            mid1_data = get_moon_position(hass, mid1, entry_id, config_data=config_data)
            mid2_data = get_moon_position(hass, mid2, entry_id, config_data=config_data)
            mid1_azimuth = normalize_azimuth(mid1_data.get('azimuth', 0))
            mid2_azimuth = normalize_azimuth(mid2_data.get('azimuth', 0))
            
            # Calculate azimuth differences (handle wrap-around)
            diff1 = abs(mid1_azimuth - target_azimuth)
            if diff1 > 180:
                diff1 = 360 - diff1
                
            diff2 = abs(mid2_azimuth - target_azimuth)
            if diff2 > 180:
                diff2 = 360 - diff2
            
            # Track best result
            if diff1 < best_azimuth_diff:
                best_azimuth_diff = diff1
                best_time = mid1
                
            if diff2 < best_azimuth_diff:
                best_azimuth_diff = diff2
                best_time = mid2
            
            # Check if we've achieved degree-based tolerance
            if abs(best_azimuth_diff) <= AZIMUTH_DEGREE_TOLERANCE:
                return best_time, iterations
            
            # Also check azimuth-based tolerance as a fallback to prevent infinite loops
            left_azimuth = normalize_azimuth(get_moon_position(hass, left, entry_id, config_data=config_data).get('azimuth', 0))
            right_azimuth = normalize_azimuth(get_moon_position(hass, right, entry_id, config_data=config_data).get('azimuth', 0))
            azimuth_window_diff = abs(right_azimuth - left_azimuth)
            if azimuth_window_diff > 180:  # Handle wraparound
                azimuth_window_diff = 360 - azimuth_window_diff
            if azimuth_window_diff <= AZIMUTH_DEGREE_TOLERANCE:
                return (best_time if best_time is not None else left + (right - left) / 2), iterations
            
            # Direction-agnostic window narrowing
            # Narrow the window toward whichever test point is closer to target
            if diff1 < diff2:
                # mid1 is closer to target, search around mid1
                # Check which half contains the target
                if is_target_in_window(left, mid1):
                    right = mid1
                elif is_target_in_window(mid1, right):
                    left = mid1
                else:
                    # Target might be very close to mid1, narrow around it
                    quarter = (right - left) / 4
                    left = mid1 - quarter
                    right = mid1 + quarter
            else:
                # mid2 is closer to target, search around mid2
                # Check which half contains the target
                if is_target_in_window(left, mid2):
                    right = mid2
                elif is_target_in_window(mid2, right):
                    left = mid2
                else:
                    # Target might be very close to mid2, narrow around it
                    quarter = (right - left) / 4
                    left = mid2 - quarter
                    right = mid2 + quarter
            
            # Ensure we don't go before current_dt (both in UTC)
            if left < current_dt_utc:
                left = current_dt_utc
        
        # Check if we hit the iteration limit
        if iterations >= AZIMUTH_TERNARY_SEARCH_MAX_ITERATIONS:
            return (best_time if best_time is not None else left + (right - left) / 2), iterations
        
        # Use best time found or middle of final window
        return (best_time if best_time is not None else left + (right - left) / 2), iterations

    # Convert search window to seconds
    initial_window_seconds = search_window_hours * 3600
    
    # Perform the ternary search
    result, iterations = ternary_search(start_dt_utc, initial_window_seconds)
    
    # Calculate performance metrics
    end_time = time.time()
    execution_time = end_time - start_time
    
    metrics = {
        'execution_time_ms': round(execution_time * 1000, 2),
        'iterations': iterations,
        'max_iterations': AZIMUTH_TERNARY_SEARCH_MAX_ITERATIONS,
        'hit_iteration_limit': iterations >= AZIMUTH_TERNARY_SEARCH_MAX_ITERATIONS,
        'target_azimuth': target_azimuth,
        'search_window_hours': search_window_hours
    }
    
    return result, metrics


# Module-level cache for azimuth reversals
_azimuth_reversal_cache = {}


def get_azimuth_reversals(
    hass: HomeAssistant,
    date: datetime.date,
    entry_id: str,
    config_data: dict = None
) -> dict:
    """
    Get cached or calculate azimuth reversals for a given date.
    
    Args:
        hass: Home Assistant instance
        date: Date to get reversals for
        entry_id: Config entry ID for location data
    
    Returns:
        Dictionary containing:
        - reversals: List of reversal points with 'time' and 'azimuth' keys
        - is_tropical: Whether location experiences azimuth reversals
        - min_azimuth: Minimum azimuth reached during the day
        - max_azimuth: Maximum azimuth reached during the day
    """
    global _azimuth_reversal_cache
    
    # Ensure this entry has a cache bucket
    if entry_id not in _azimuth_reversal_cache:
        _azimuth_reversal_cache[entry_id] = {}
    
    entry_cache = _azimuth_reversal_cache[entry_id]
    
    # Clean old cache entries at local midnight (only for this entry)
    import zoneinfo
    local_tz = zoneinfo.ZoneInfo(hass.config.time_zone)
    current_local_time = datetime.datetime.now(local_tz)
    current_local_date = current_local_time.date()
    
    keys_to_remove = []
    for cache_key in entry_cache.keys():
        cache_date = datetime.datetime.strptime(cache_key, '%Y-%m-%d').date()
        if cache_date < current_local_date:
            keys_to_remove.append(cache_key)
    
    for key in keys_to_remove:
        del entry_cache[key]
    
    result = None
    target_dates = [date]
    if date == current_local_date:
        target_dates.append(date + datetime.timedelta(days=1))

    for target_date in target_dates:
        date_key = target_date.strftime('%Y-%m-%d')

        if date_key not in entry_cache:
            entry_cache[date_key] = _calculate_azimuth_reversals(
                hass, target_date, entry_id, config_data=config_data
            )

        if target_date == date:
            result = entry_cache[date_key]

    return result


def get_current_azimuth_direction(
    current_time: datetime.datetime,
    base_direction: int,
    reversals: list
) -> int:
    """
    Determine current azimuth direction based on base direction and reversal history.
    
    Args:
        current_time: Current datetime
        base_direction: Base direction from midnight (1 for positive, -1 for negative)
        reversals: List of reversal points with 'time' and 'azimuth' keys
    
    Returns:
        Current azimuth direction (1 for positive, -1 for negative)
    """
    # Start with base direction (from midnight)
    current_direction = base_direction
    
    # Apply each reversal that occurred before current_time
    for reversal in sorted(reversals, key=lambda r: r['time']):
        if reversal['time'] <= current_time:
            # Each reversal flips the direction
            current_direction *= -1
        else:
            break  # No more past reversals
    
    return current_direction


def _calculate_azimuth_reversals(
    hass: HomeAssistant,
    date: datetime.date,
    entry_id: str,
    config_data: dict = None
) -> dict:
    """
    Calculate all azimuth reversals for a given date using hybrid linear/binary search.
    Always scans from local midnight to local midnight to ensure complete cache data.
    
    Args:
        hass: Home Assistant instance
        date: Date to calculate reversals for (local date)
        entry_id: Config entry ID for location data
    
    Returns:
        Dictionary with reversal data including base direction from local midnight
    """
    # Get config data if not provided
    if config_data is None:
        from .config_store import get_config_entry_data
        config_data = get_config_entry_data(entry_id)
    
    from datetime import time, timedelta
    
    # Get timezone for this location
    import zoneinfo
    local_tz = zoneinfo.ZoneInfo(hass.config.time_zone)
    
    # Helper function to calculate azimuth derivative with wrap-around handling
    def calculate_azimuth_derivative(az1: float, az2: float) -> float:
        """Calculate azimuth change handling wrap-around."""
        diff = az2 - az1
        
        # Handle wrap-around cases
        if diff > 180:
            diff -= 360  # e.g., 10° - 350° = 20° (not -340°)
        elif diff < -180:
            diff += 360  # e.g., 350° - 10° = -20° (not 340°)
        
        return diff
    
    # Calculate base direction from local midnight
    def get_base_direction_from_midnight() -> int:
        """Determine base azimuth direction by sampling around local midnight."""
        # Create local midnight for the given date, then convert to UTC
        local_midnight = datetime.datetime.combine(date, time(0, 0, 0), tzinfo=local_tz)
        midnight_utc = local_midnight.astimezone(timezone.utc)
        
        # Sample azimuth change around midnight (avoid exactly midnight to prevent edge cases)
        time_before = midnight_utc + timedelta(minutes=1)
        time_after = midnight_utc + timedelta(minutes=6)
        
        az_before = get_moon_position(hass, time_before, entry_id, config_data=config_data)['azimuth']
        az_after = get_moon_position(hass, time_after, entry_id, config_data=config_data)['azimuth']
        
        # Calculate direction of change
        derivative = calculate_azimuth_derivative(az_before, az_after)
        
        return 1 if derivative > 0 else -1
    
    # Helper function for binary search refinement
    def binary_search_reversal(start_time: datetime.datetime, end_time: datetime.datetime) -> tuple[datetime.datetime, float]:
        """Binary search to find exact reversal point within tolerance."""
        left, right = start_time, end_time
        iterations = 0
        
        while iterations < AZIMUTH_REVERSAL_SEARCH_MAX_ITERATIONS:
            # Check azimuth difference across current window
            left_azimuth = get_moon_position(hass, left, entry_id, config_data=config_data)['azimuth']
            right_azimuth = get_moon_position(hass, right, entry_id, config_data=config_data)['azimuth']
            azimuth_diff = abs(right_azimuth - left_azimuth)
            if azimuth_diff > 180:  # Handle wraparound
                azimuth_diff = 360 - azimuth_diff
            
            # Stop if azimuth difference is small enough
            if azimuth_diff <= AZIMUTH_DEGREE_TOLERANCE:
                break
                
            iterations += 1
            mid = left + (right - left) / 2
            
            # Check if reversal is in left or right half
            left_mid = left + (mid - left) / 2
            right_mid = mid + (right - mid) / 2
            
            # Get azimuth values
            left_az = get_moon_position(hass, left, entry_id, config_data=config_data)['azimuth']
            left_mid_az = get_moon_position(hass, left_mid, entry_id, config_data=config_data)['azimuth']
            right_mid_az = get_moon_position(hass, right_mid, entry_id, config_data=config_data)['azimuth']
            right_az = get_moon_position(hass, right, entry_id, config_data=config_data)['azimuth']
            
            # Calculate derivatives
            left_rate = calculate_azimuth_derivative(left_az, left_mid_az)
            right_rate = calculate_azimuth_derivative(right_mid_az, right_az)
            
            # Check which half contains the reversal
            if left_rate * right_rate < 0:
                # Reversal is in the right half
                left = left_mid
            else:
                # Reversal is in the left half
                right = right_mid
        
        # Return the precise time and azimuth
        final_time = left + (right - left) / 2
        final_azimuth = get_moon_position(hass, final_time, entry_id, config_data=config_data)['azimuth']
        return final_time, final_azimuth
    
    # Create local midnight boundaries and convert to UTC for calculations
    local_midnight_start = datetime.datetime.combine(date, time(0, 0, 0), tzinfo=local_tz)
    local_midnight_end = local_midnight_start + timedelta(days=1)
    
    # Convert to UTC for all calculations
    start_time = local_midnight_start.astimezone(timezone.utc)
    end_time = local_midnight_end.astimezone(timezone.utc)
    
    # Calculate base direction from local midnight
    base_direction = get_base_direction_from_midnight()
    
    # Phase 1: Linear scan every 5 minutes for 24 hours (in UTC)
    candidate_reversals = []
    current_time = start_time + timedelta(minutes=5)  # Start 5 minutes in to have a previous point
    
    min_azimuth = float('inf')
    max_azimuth = float('-inf')
    
    while current_time < end_time - timedelta(minutes=5):  # End 5 minutes early to have a next point
        prev_time = current_time - timedelta(minutes=5)
        next_time = current_time + timedelta(minutes=5)
        
        # Get azimuth values (all times are in UTC)
        prev_az = get_moon_position(hass, prev_time, entry_id, config_data=config_data)['azimuth']
        curr_az = get_moon_position(hass, current_time, entry_id, config_data=config_data)['azimuth']
        next_az = get_moon_position(hass, next_time, entry_id, config_data=config_data)['azimuth']
        
        # Track min/max azimuth
        min_azimuth = min(min_azimuth, curr_az)
        max_azimuth = max(max_azimuth, curr_az)
        
        # Calculate derivatives
        rate_before = calculate_azimuth_derivative(prev_az, curr_az)
        rate_after = calculate_azimuth_derivative(curr_az, next_az)
        
        # Check for sign change (reversal)
        if rate_before * rate_after < 0:
            candidate_reversals.append((prev_time, next_time))
        
        current_time += timedelta(minutes=5)
    
    # Phase 2: Binary search to refine each candidate to precise tolerance
    precise_reversals = []
    for start_window, end_window in candidate_reversals:
        try:
            precise_time, precise_azimuth = binary_search_reversal(start_window, end_window)
            precise_reversals.append({
                'time': precise_time,  # This will be in UTC
                'azimuth': precise_azimuth
            })
        except Exception as e:
            continue
    
    # Determine if location is tropical (has reversals)
    is_tropical = len(precise_reversals) > 0
    
    return {
        'base_direction': base_direction,
        'reversals': precise_reversals,  # All times in UTC
        'is_tropical': is_tropical,
        'min_azimuth': min_azimuth if min_azimuth != float('inf') else 0,
        'max_azimuth': max_azimuth if max_azimuth != float('-inf') else 360
    }

