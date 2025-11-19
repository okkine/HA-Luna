"""Sensors for Luna integration."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import DOMAIN, DEBUG_ELEVATION_SENSOR, DEBUG_AZIMUTH_SENSOR, DEBUG_ATTRIBUTES
from .base_sensors import BaseElevationSensor, BaseAzimuthSensor, BasePositionSensor
from .utils import get_moon_position, format_sensor_naming, get_moon_phase, get_moonrise_moonset_for_date
from .config_store import get_config_entry_data
from .utils import get_next_step, get_time_at_elevation, get_time_at_azimuth, get_azimuth_reversals
from .cache import get_cached_lunar_event

_LOGGER = logging.getLogger(__name__)


def _format_duration_hhmmss(seconds: float | None) -> str | None:
    """Format duration in seconds to HH:MM:SS string."""
    if seconds is None:
        return None
    # total_seconds() returns seconds, not milliseconds
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Initialize Luna config entry."""
    
    
    # Get configuration data
    config_data = get_config_entry_data(config_entry.entry_id)
    
    
    # Create sensors list with exception handling
    sensors = []
    
    try:
        
        elevation_sensor = ElevationSensor(hass, config_entry.entry_id)
        sensors.append(elevation_sensor)
        
    except Exception as e:
        _LOGGER.error(f"Failed to create ElevationSensor for entry {config_entry.entry_id}: {e}")
    
    try:
        
        azimuth_sensor = AzimuthSensor(hass, config_entry.entry_id)
        sensors.append(azimuth_sensor)
        
    except Exception as e:
        _LOGGER.error(f"Failed to create AzimuthSensor for entry {config_entry.entry_id}: {e}")
    
    try:
        
        percent_illuminated_sensor = PercentIlluminatedSensor(hass, config_entry.entry_id)
        sensors.append(percent_illuminated_sensor)
        
    except Exception as e:
        _LOGGER.error(f"Failed to create PercentIlluminatedSensor for entry {config_entry.entry_id}: {e}")
    
    try:
        
        phase_percent_sensor = LunarPhasePercentSensor(hass, config_entry.entry_id)
        sensors.append(phase_percent_sensor)
        
    except Exception as e:
        _LOGGER.error(f"Failed to create LunarPhasePercentSensor for entry {config_entry.entry_id}: {e}")
    
    try:
        
        phase_name_sensor = LunarPhaseNameSensor(hass, config_entry.entry_id)
        sensors.append(phase_name_sensor)
        
    except Exception as e:
        _LOGGER.error(f"Failed to create LunarPhaseNameSensor for entry {config_entry.entry_id}: {e}")
    
    try:
        
        phase_degrees_sensor = LunarPhaseDegreesSensor(hass, config_entry.entry_id)
        sensors.append(phase_degrees_sensor)
        
    except Exception as e:
        _LOGGER.error(f"Failed to create LunarPhaseDegreesSensor for entry {config_entry.entry_id}: {e}")
    
    try:
        
        moonrise_sensor = MoonriseSensor(hass, config_entry.entry_id)
        sensors.append(moonrise_sensor)
        
    except Exception as e:
        _LOGGER.error(f"Failed to create MoonriseSensor for entry {config_entry.entry_id}: {e}")
    
    try:
        
        moonset_sensor = MoonsetSensor(hass, config_entry.entry_id)
        sensors.append(moonset_sensor)
        
    except Exception as e:
        _LOGGER.error(f"Failed to create MoonsetSensor for entry {config_entry.entry_id}: {e}")
    
    
    
    if sensors:
        try:
            
            async_add_entities(sensors)
            
        except Exception as e:
            _LOGGER.error(f"Failed to add sensors to Home Assistant for entry {config_entry.entry_id}: {e}")
    else:
        _LOGGER.error(f"No sensors were created for entry {config_entry.entry_id}")


class ElevationSensor(BaseElevationSensor):
    """Sensor for moon elevation."""
    def __init__(self, hass: HomeAssistant, entry_id: str):
        try:
            
            # Format the name and get unique ID
            name = "Lunar Elevation"
            formatted_name, unique_id = format_sensor_naming(name, entry_id)
      
            # Pass the formatted name and unique_id
            super().__init__(hass=hass, sensor_name=formatted_name, unique_id=unique_id)
            
            # Store entry_id for moon position calculations
            self._entry_id = entry_id
            
            # Add device info to link this sensor to the device
            self._attr_device_info = {
                "identifiers": {(DOMAIN, entry_id)},
            }
            
            
            
        except Exception as e:
            _LOGGER.error(f"Failed to create ElevationSensor for entry {entry_id}: {e}", exc_info=True)
            raise

    async def _async_update_logic(self, now=None) -> datetime:
        """Update the sensor value and return next update time."""
        try:
            # Use current time if not provided
            if now is None:
                now = datetime.now()
            
            # Get elevation step from config entry
            config_data = get_config_entry_data(self._entry_id)
            elevation_step = config_data.get('elevation_step', 0.5) if config_data else 0.5
            
            # Get current moon position data
            moon_data = get_moon_position(self.hass, now, self._entry_id, config_data=config_data)
            
            # Update the sensor value
            self._attr_native_value = round(moon_data['elevation'], 2)
            
            # Calculate the next elevation step
            current_elevation = moon_data['elevation']
            next_elevation = get_next_step(
                'elevation',
                elevation_step,
                moon_data,
                now,
                self._entry_id,
                self.hass,
                DEBUG_ELEVATION_SENSOR
            )
            # Only pass a float to get_time_at_elevation; allow override time for event cases
            next_time_override = None
            if isinstance(next_elevation, dict):
                elevation_target = float(next_elevation.get('elevation', 0.0) or 0.0)
                next_time_override = next_elevation.get('next_time')
            elif next_elevation is None:
                elevation_target = 0.0
            else:
                elevation_target = float(next_elevation)

            if next_time_override:
                # Use cached UTC event time directly to avoid tangent search
                next_update_time = next_time_override + timedelta(milliseconds=100)
            else:
                # Get the time when moon will be at the next elevation step
                next_rising, next_setting, next_event = get_time_at_elevation(
                    self.hass, elevation_target, now, self._entry_id, next_transit_fallback=True, config_data=config_data
                )
                
                # Use next_event as the next update time (handles all edge cases)
                if next_event:
                    next_update_time = next_event
                else:
                    # Fallback to 60 seconds if no event found
                    next_update_time = datetime.now(timezone.utc) + timedelta(seconds=60)
            
            # Use the target elevation as next_target
            next_target = next_elevation
            
            # Get lunar events cache information
            try:
                cached_next_event_time, cached_next_event_type, cached_next_event_elevation, used_cache = get_cached_lunar_event(
                    self.hass, self._entry_id, now
                )
                
                # Get cache information for debugging
                from .cache import get_lunar_events_cache_instance
                lunar_events_cache_instance = get_lunar_events_cache_instance(self.hass)
                entry_cache = lunar_events_cache_instance._get_entry_cache(self._entry_id)
                
                lunar_events_cache = {
                    'next_event_time': cached_next_event_time.isoformat() if cached_next_event_time else None,
                    'next_event_type': cached_next_event_type,
                    'next_event_elevation': cached_next_event_elevation,
                    'cache_available': cached_next_event_time is not None,
                    'debug_current_time': now.isoformat() if now else None,
                    'debug_timezone': str(now.tzinfo) if now and now.tzinfo else None,
                    'debug_sensor_entry_id': self._entry_id,
                    'debug_used_cache': used_cache
                }
            except Exception as e:
                # Get cache information for debugging even in error case
                from .cache import get_lunar_events_cache_instance
                lunar_events_cache_instance = get_lunar_events_cache_instance(self.hass)
                entry_cache = lunar_events_cache_instance._get_entry_cache(self._entry_id)
                
                lunar_events_cache = {
                    'next_event_time': None,
                    'next_event_type': None,
                    'next_event_elevation': None,
                    'cache_available': False,
                    'cache_error': str(e),
                    'debug_current_time': now.isoformat() if now else None,
                    'debug_timezone': str(now.tzinfo) if now and now.tzinfo else None,
                    'debug_sensor_entry_id': self._entry_id,
                    'debug_used_cache': False
                }
            
            # Store all moon position data as attributes
            # Handle next_target which could be a dict, float, or None
            if isinstance(next_target, dict):
                next_target_elevation = next_target.get('elevation', 'Unknown')
                next_target_event = next_target.get('event', 'Unknown')
            elif next_target is None:
                next_target_elevation = 'Unknown'
                next_target_event = 'Unknown'
            else:
                # It's a float
                next_target_elevation = next_target
                next_target_event = 'step'
            
            attributes = {
                'next_update': next_update_time,
                'next_target': next_target_elevation,
            }
            
            if DEBUG_ATTRIBUTES:
                attributes['moon_data'] = moon_data
                attributes['lunar_events_cache'] = lunar_events_cache
                attributes['next_event'] = next_target_event
                attributes['size'] = moon_data['size']
                attributes['declination'] = moon_data['declination']
                attributes['latitude'] = moon_data['latitude']
                attributes['longitude'] = moon_data['longitude']
                attributes['elevation_m'] = moon_data['elevation_m']
                attributes['pressure_mbar'] = moon_data['pressure_mbar']
                attributes['phase_percent'] = moon_data.get('phase_percent')
                attributes['previous_new_moon'] = moon_data.get('previous_new_moon')
                attributes['next_new_moon'] = moon_data.get('next_new_moon')
            self._attr_extra_state_attributes = attributes
            
            return next_update_time
            
        except Exception as e:
            # On error, retry in 60 seconds
            _LOGGER.error(f"Error updating ElevationSensor for entry {self._entry_id}: {e}", exc_info=True)
            return datetime.now(timezone.utc) + timedelta(seconds=60)


class AzimuthSensor(BaseAzimuthSensor):
    """Sensor for moon azimuth."""
    def __init__(self, hass: HomeAssistant, entry_id: str):
        try:
            
            # Format the name
            name = "Lunar Azimuth"
            formatted_name, unique_id = format_sensor_naming(name, entry_id)

            # Pass the formatted name and unique_id
            super().__init__(hass=hass, sensor_name=formatted_name, unique_id=unique_id)
            
            # Store entry_id for moon position calculations
            self._entry_id = entry_id
            
            # Add device info to link this sensor to the device
            self._attr_device_info = {
                "identifiers": {(DOMAIN, entry_id)},
            }
            
            
            
        except Exception as e:
            _LOGGER.error(f"Failed to create AzimuthSensor for entry {entry_id}: {e}", exc_info=True)
            raise

    async def _async_update_logic(self, now=None) -> datetime:
        """Update the sensor value and return next update time."""
        
        try:
            
            # Use current time if not provided
            if now is None:
                import zoneinfo
                tz = zoneinfo.ZoneInfo(self.hass.config.time_zone)
                now = datetime.now(tz)
            
            
            # Get azimuth step from config entry
            config_data = get_config_entry_data(self._entry_id)
            azimuth_step = config_data.get('azimuth_step', 1.0) if config_data else 1.0
            
            # Get current moon position data
            moon_data = get_moon_position(self.hass, now, self._entry_id, config_data=config_data)
            
            # Update the sensor value
            self._attr_native_value = round(moon_data['azimuth'], 2) % 360
            
            # Calculate the next azimuth step using enhanced logic with reversal cache
            next_step_info = get_next_step(
                'azimuth',
                azimuth_step,
                moon_data,
                now,
                self._entry_id,
                self.hass,
                DEBUG_AZIMUTH_SENSOR,
                config_data=config_data
            )
            if isinstance(next_step_info, dict):
                next_azimuth_target = next_step_info["azimuth"]
                is_reversal = next_step_info.get("reversal", False)
                reversal_time = next_step_info.get("reversal_time")
            else:
                next_azimuth_target = next_step_info
                is_reversal = False
                reversal_time = None

            calculated_direction = 0
            try:
                current_date = now.date() if now else datetime.now().date()
                temp_reversal_data = get_azimuth_reversals(self.hass, current_date, self._entry_id, config_data=config_data)
                reversals = temp_reversal_data.get('reversals', [])
                base_direction = temp_reversal_data.get('base_direction', 1)
                now_utc = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
                calculated_direction = base_direction
                for reversal in sorted(reversals, key=lambda r: r['time']):
                    if reversal['time'] <= now_utc:
                        calculated_direction *= -1
                    else:
                        break
            except Exception as e:
                calculated_direction = f"Error: {e}"

            if is_reversal and reversal_time is not None:
                next_update_time = reversal_time
                search_metrics = {"info": "Next target is a reversal; using cached reversal time."}
            else:
                # Only pass a float to get_time_at_azimuth
                if isinstance(next_azimuth_target, dict):
                    az_target = float(next_azimuth_target.get('azimuth', 0.0) or 0.0)
                elif next_azimuth_target is None:
                    az_target = 0.0
                else:
                    az_target = float(next_azimuth_target)
                next_update_time, search_metrics = get_time_at_azimuth(
                    self.hass,
                    az_target,
                    now,
                    self._entry_id,
                    None,  # start_dt
                    0.167,  # search_window_hours (10 minutes)
                    config_data=config_data
                )
            
            # Handle calculation failure
            if next_update_time is None:
                # Let it fail naturally so we can debug the issue
                next_azimuth_target = None
            
        except Exception as e:
            # Let the exception propagate so we can debug it
            raise
        
        # Get reversal cache data for today (separate try/catch to ensure attributes are always set)
        try:
            current_date = now.date() if now else datetime.now().date()
            reversal_data = get_azimuth_reversals(self.hass, current_date, self._entry_id, config_data=config_data)
        except Exception as e:
            reversal_data = {
                'reversals': [],
                'is_tropical': False,
                'min_azimuth': 0,
                'max_azimuth': 360,
                'error': str(e)
            }
        
        # Store all moon position data as attributes, including next target and update time
        # This will ALWAYS execute regardless of any exceptions above
        try:
            attributes = {
                'next_update': next_update_time,
                'next_target': next_azimuth_target,
            }
            
            if DEBUG_ATTRIBUTES:
                attributes['elevation'] = moon_data['elevation']
                attributes['declination'] = moon_data['declination']
                attributes['size'] = moon_data['size']
                attributes['latitude'] = moon_data['latitude']
                attributes['longitude'] = moon_data['longitude']
                attributes['elevation_m'] = moon_data['elevation_m']
                attributes['pressure_mbar'] = moon_data['pressure_mbar']
                attributes['phase_percent'] = moon_data.get('phase_percent')
                attributes['previous_new_moon'] = moon_data.get('previous_new_moon')
                attributes['next_new_moon'] = moon_data.get('next_new_moon')
                attributes['reversal'] = is_reversal
                attributes['reversal_time'] = reversal_time
                attributes['search_performance'] = search_metrics
                attributes['reversal_cache'] = reversal_data
                attributes['calculated_direction'] = calculated_direction
            
            self._attr_extra_state_attributes = attributes
        except Exception as e:
            # Minimal attributes if even this fails
            self._attr_extra_state_attributes = {
                'error': f'Attribute error: {e}',
                'reversal_cache': reversal_data
            }
        
        # Ensure we always return a datetime, not None
        if next_update_time is None:
            return datetime.now(timezone.utc) + timedelta(seconds=60)
        
        return next_update_time


class PercentIlluminatedSensor(BasePositionSensor):
    """Sensor for moon percent illuminated, updated via events from elevation/azimuth sensors."""
    
    def __init__(self, hass: HomeAssistant, entry_id: str):
        try:
            # Format the name and get unique ID
            name = "Lunar Percent Illuminated"
            formatted_name, unique_id = format_sensor_naming(name, entry_id)
            
            # Pass the formatted name and unique_id
            super().__init__(hass=hass, sensor_name=formatted_name, unique_id=unique_id)
            
            # Store entry_id for moon position calculations
            self._entry_id = entry_id
            
            # Add device info to link this sensor to the device
            self._attr_device_info = {
                "identifiers": {(DOMAIN, entry_id)},
            }
            
            # Set sensor properties
            self._attr_device_class = None
            self._attr_state_class = "measurement"
            
            # Initialize phase_percent for icon calculation
            self._phase_percent = None
            
        except Exception as e:
            _LOGGER.error(f"Failed to create PercentIlluminatedSensor for entry {entry_id}: {e}", exc_info=True)
            raise
    
    @property
    def native_unit_of_measurement(self):
        """Return the unit of measurement."""
        return "%"
    
    @property
    def icon(self):
        """Return the icon based on moon phase."""
        # Calculate current phase on-demand for real-time icon updates
        try:
            now = datetime.now(timezone.utc)
            config_data = get_config_entry_data(self._entry_id)
            moon_data = get_moon_position(self.hass, now, self._entry_id, config_data=config_data)
            current_phase_percent = moon_data.get("phase_percent")
            _, icon_name = get_moon_phase(current_phase_percent)
            return icon_name
        except Exception:
            # Fallback to cached value if calculation fails
            _, icon_name = get_moon_phase(self._phase_percent)
            return icon_name
    
    async def _async_update_logic(self, now=None) -> datetime | None:
        """Update percent illuminated using current moon position data."""
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        config_data = get_config_entry_data(self._entry_id)
        moon_data = get_moon_position(self.hass, now, self._entry_id, config_data=config_data)

        percent_illuminated = moon_data.get("percent_illuminated")
        phase_percent = moon_data.get("phase_percent")

        self._phase_percent = phase_percent
        self._attr_native_value = round(percent_illuminated, 1) if percent_illuminated is not None else None
        self._attr_available = percent_illuminated is not None

        # Calculate next update time at next hour
        next_update = self._get_next_hour(now)
        
        attributes = {
            "next_update": next_update,
        }
        
        if DEBUG_ATTRIBUTES:
            attributes["phase_percent"] = phase_percent
        
        self._attr_extra_state_attributes = attributes

        return next_update

    def _get_next_hour(self, current_time: datetime) -> datetime:
        """Return the next top-of-hour UTC datetime."""
        next_hour = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return next_hour


class LunarPhasePercentSensor(BasePositionSensor):
    """Sensor for moon phase percent (position in lunar cycle), updated hourly on the hour."""

    def __init__(self, hass: HomeAssistant, entry_id: str):
        try:
            name = "Lunar Phase Percent"
            formatted_name, unique_id = format_sensor_naming(name, entry_id)

            super().__init__(hass=hass, sensor_name=formatted_name, unique_id=unique_id)

            self._entry_id = entry_id
            self._attr_device_info = {
                "identifiers": {(DOMAIN, entry_id)},
            }
            self._attr_device_class = None
            self._attr_state_class = "measurement"
            self._phase_percent = None
        except Exception as e:
            _LOGGER.error(f"Failed to create LunarPhasePercentSensor for entry {entry_id}: {e}", exc_info=True)
            raise

    @property
    def native_unit_of_measurement(self):
        return "%"

    @property
    def icon(self):
        """Return the icon based on moon phase."""
        # Calculate current phase on-demand for real-time icon updates
        try:
            now = datetime.now(timezone.utc)
            config_data = get_config_entry_data(self._entry_id)
            moon_data = get_moon_position(self.hass, now, self._entry_id, config_data=config_data)
            current_phase_percent = moon_data.get("phase_percent")
            _, icon_name = get_moon_phase(current_phase_percent)
            return icon_name
        except Exception:
            # Fallback to cached value if calculation fails
            _, icon_name = get_moon_phase(self._phase_percent)
            return icon_name

    async def _async_update_logic(self, now=None) -> datetime | None:
        """Update phase percent using the current lunar cycle data."""
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        config_data = get_config_entry_data(self._entry_id)
        moon_data = get_moon_position(self.hass, now, self._entry_id, config_data=config_data)

        phase_percent = moon_data.get("phase_percent")
        previous_new_moon = moon_data.get("previous_new_moon")
        next_new_moon = moon_data.get("next_new_moon")

        self._phase_percent = phase_percent
        
        # Round to nearest integer for sensor value, wrapping 100% to 0%
        if phase_percent is not None:
            rounded_percent = round(phase_percent) % 100
        else:
            rounded_percent = None
        self._attr_native_value = rounded_percent
        self._attr_available = phase_percent is not None

        # Calculate next update time at next 1% increment
        percent_length_seconds = None
        next_update = None
        next_percent = None
        if phase_percent is not None and previous_new_moon and next_new_moon:
            # Calculate time per percent
            cycle_duration = (next_new_moon - previous_new_moon).total_seconds()
            percent_length_seconds = cycle_duration / 100.0
            
            # Find next integer percent based on current sensor value
            # Use rounded value to ensure consistency
            next_percent = rounded_percent + 1
            if next_percent >= 100:
                # Wrap around: next percent is 0, which is the next new moon
                next_percent = 0
                next_update = next_new_moon + timedelta(seconds=1)
            else:
                # Calculate absolute time when moon will be at next_percent
                # Time from previous_new_moon to next_percent
                time_to_next_percent = next_percent * percent_length_seconds
                next_update = previous_new_moon + timedelta(seconds=time_to_next_percent + 1)
        else:
            # Fallback to hourly if we don't have the data
            next_update = self._get_next_hour(now)
        
        attributes = {
            "next_update": next_update,
            "next_target": next_percent,
        }
        
        if DEBUG_ATTRIBUTES:
            attributes["previous_new_moon"] = previous_new_moon
            attributes["next_new_moon"] = next_new_moon
            attributes["percent_length"] = _format_duration_hhmmss(percent_length_seconds)
        
        self._attr_extra_state_attributes = attributes

        return next_update

    def _get_next_hour(self, current_time: datetime) -> datetime:
        """Return the next top-of-hour UTC datetime."""
        next_hour = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return next_hour


class LunarPhaseNameSensor(BasePositionSensor):
    """Sensor for moon phase name (string), updated at phase threshold boundaries."""
    
    def __init__(self, hass: HomeAssistant, entry_id: str):
        try:
            # Format the name and get unique ID
            name = "Lunar Phase"
            formatted_name, unique_id = format_sensor_naming(name, entry_id)
            
            # Pass the formatted name and unique_id
            super().__init__(hass=hass, sensor_name=formatted_name, unique_id=unique_id)
            
            # Store entry_id
            self._entry_id = entry_id
            
            # Add device info to link this sensor to the device
            self._attr_device_info = {
                "identifiers": {(DOMAIN, entry_id)},
            }
            
            # Set sensor properties
            self._attr_device_class = None
            self._attr_state_class = None  # No state class for string sensors
            
            # Initialize phase_percent for icon calculation
            self._phase_percent = None
            
        except Exception as e:
            _LOGGER.error(f"Failed to create LunarPhaseNameSensor for entry {entry_id}: {e}", exc_info=True)
            raise
    
    @property
    def native_unit_of_measurement(self):
        """Return None for string sensors."""
        return None
    
    @property
    def icon(self):
        """Return the icon based on moon phase."""
        # Calculate current phase on-demand for real-time icon updates
        try:
            now = datetime.now(timezone.utc)
            config_data = get_config_entry_data(self._entry_id)
            moon_data = get_moon_position(self.hass, now, self._entry_id, config_data=config_data)
            current_phase_percent = moon_data.get("phase_percent")
            _, icon_name = get_moon_phase(current_phase_percent)
            return icon_name
        except Exception:
            # Fallback to cached value if calculation fails
            _, icon_name = get_moon_phase(self._phase_percent)
            return icon_name
    
    def _get_next_phase_threshold(self, phase_percent: float) -> float:
        """
        Get the next phase threshold boundary based on current phase_percent.
        
        Phase thresholds:
        - New Moon: <= 1 or >= 98 → next: 2 (if <= 1) or 1 (if >= 98, wrap)
        - Waxing Crescent: < 23 → next: 23
        - First Quarter: >= 23 and <= 26 → next: 27
        - Waxing Gibbous: < 48 → next: 48
        - Full Moon: >= 48 and <= 51 → next: 52
        - Waning Gibbous: < 73 → next: 73
        - Last Quarter: >= 73 and <= 76 → next: 77
        - Waning Crescent: >= 77 → next: 98 (wrap to New Moon)
        """
        rounded_phase = round(phase_percent)
        
        if rounded_phase <= 1:
            return 2.0  # Next: Waxing Crescent
        elif rounded_phase >= 98:
            return 1.0  # Next: New Moon (wrap around)
        elif rounded_phase < 23:
            return 23.0  # Next: First Quarter
        elif rounded_phase >= 23 and rounded_phase <= 26:
            return 27.0  # Next: Waxing Gibbous
        elif rounded_phase < 48:
            return 48.0  # Next: Full Moon
        elif rounded_phase >= 48 and rounded_phase <= 51:
            return 52.0  # Next: Waning Gibbous
        elif rounded_phase < 73:
            return 73.0  # Next: Last Quarter
        elif rounded_phase >= 73 and rounded_phase <= 76:
            return 77.0  # Next: Waning Crescent
        else:  # 77-97: Waning Crescent
            return 98.0  # Next: New Moon (wrap around)
    
    def _get_next_phase_threshold_from_name(self, current_phase_name: str, phase_percent: float) -> float:
        """
        Get the next phase threshold boundary based on current phase name.
        This ensures we get the correct next phase even when at threshold boundaries.
        
        Args:
            current_phase_name: Current phase name (e.g., "New Moon", "Waxing Crescent")
            phase_percent: Current phase percent (used for wrap-around detection)
        
        Returns:
            Next phase threshold percentage
        """
        if current_phase_name == "New Moon":
            # New Moon can be at 0-1% or 98-100%
            # If we're near the end (>= 98), next is 2% (Waxing Crescent) after wrapping
            # If we're at the start (<= 1), next is 2% (Waxing Crescent)
            # In both cases, next is 2%
            return 2.0  # Next: Waxing Crescent
        elif current_phase_name == "Waxing Crescent":
            return 23.0  # Next: First Quarter
        elif current_phase_name == "First Quarter":
            return 27.0  # Next: Waxing Gibbous
        elif current_phase_name == "Waxing Gibbous":
            return 48.0  # Next: Full Moon
        elif current_phase_name == "Full Moon":
            return 52.0  # Next: Waning Gibbous
        elif current_phase_name == "Waning Gibbous":
            return 73.0  # Next: Last Quarter
        elif current_phase_name == "Last Quarter":
            return 77.0  # Next: Waning Crescent
        elif current_phase_name == "Waning Crescent":
            return 98.0  # Next: New Moon (wrap around)
        else:
            # Fallback to old method if phase name is unknown
            return self._get_next_phase_threshold(phase_percent)
    
    async def _async_update_logic(self, now=None) -> datetime | None:
        """Update phase name using the current lunar cycle data."""
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        config_data = get_config_entry_data(self._entry_id)
        moon_data = get_moon_position(self.hass, now, self._entry_id, config_data=config_data)

        phase_percent = moon_data.get("phase_percent")
        previous_new_moon = moon_data.get("previous_new_moon")
        next_new_moon = moon_data.get("next_new_moon")

        self._phase_percent = phase_percent
        
        # Get current phase name
        phase_name, _ = get_moon_phase(phase_percent)
        self._attr_native_value = phase_name
        self._attr_available = phase_percent is not None

        # Calculate next update time at next phase threshold
        next_update = None
        next_target_phase_name = None
        if phase_percent is not None and previous_new_moon and next_new_moon:
            # Calculate time per percent (same method as phase_percent sensor)
            cycle_duration = (next_new_moon - previous_new_moon).total_seconds()
            percent_length_seconds = cycle_duration / 100.0
            
            # Find next phase threshold based on current phase name
            # This ensures we get the correct next phase even if we're at a threshold boundary
            next_threshold = self._get_next_phase_threshold_from_name(phase_name, phase_percent)
            
            # Get the phase name for the next threshold
            next_target_phase_name, _ = get_moon_phase(next_threshold)
            
            # Calculate absolute time when moon will be at next_threshold
            # Handle wrap-around: if next_threshold < phase_percent, we're wrapping to next cycle
            if next_threshold < phase_percent:
                # Wrapping around: next threshold is in the next cycle
                # Time from next_new_moon to next_threshold
                time_to_next_threshold = next_threshold * percent_length_seconds
                next_update = next_new_moon + timedelta(seconds=time_to_next_threshold + 1)
            else:
                # Calculate absolute time from previous_new_moon to next_threshold
                time_to_next_threshold = next_threshold * percent_length_seconds
                next_update = previous_new_moon + timedelta(seconds=time_to_next_threshold + 1)
        else:
            # Fallback to hourly if we don't have the data
            next_update = self._get_next_hour(now)
        
        attributes = {
            "next_update": next_update,
            "next_target": next_target_phase_name,
        }
        
        if DEBUG_ATTRIBUTES:
            attributes["previous_new_moon"] = previous_new_moon
            attributes["next_new_moon"] = next_new_moon
        
        self._attr_extra_state_attributes = attributes

        return next_update

    def _get_next_hour(self, current_time: datetime) -> datetime:
        """Return the next top-of-hour UTC datetime."""
        next_hour = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return next_hour


class LunarPhaseDegreesSensor(BasePositionSensor):
    """Sensor for moon phase degrees (0-359), updated hourly on the hour."""

    def __init__(self, hass: HomeAssistant, entry_id: str):
        try:
            name = "Lunar Phase Degrees"
            formatted_name, unique_id = format_sensor_naming(name, entry_id)

            super().__init__(hass=hass, sensor_name=formatted_name, unique_id=unique_id)

            self._entry_id = entry_id
            self._attr_device_info = {
                "identifiers": {(DOMAIN, entry_id)},
            }
            self._attr_device_class = None
            self._attr_state_class = "measurement"
            self._phase_percent = None
        except Exception as e:
            _LOGGER.error(f"Failed to create LunarPhaseDegreesSensor for entry {entry_id}: {e}", exc_info=True)
            raise

    @property
    def native_unit_of_measurement(self):
        return "°"

    @property
    def icon(self):
        """Return the icon based on moon phase."""
        # Calculate current phase on-demand for real-time icon updates
        try:
            now = datetime.now(timezone.utc)
            config_data = get_config_entry_data(self._entry_id)
            moon_data = get_moon_position(self.hass, now, self._entry_id, config_data=config_data)
            current_phase_percent = moon_data.get("phase_percent")
            _, icon_name = get_moon_phase(current_phase_percent)
            return icon_name
        except Exception:
            # Fallback to cached value if calculation fails
            _, icon_name = get_moon_phase(self._phase_percent)
            return icon_name

    async def _async_update_logic(self, now=None) -> datetime | None:
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        config_data = get_config_entry_data(self._entry_id)
        moon_data = get_moon_position(self.hass, now, self._entry_id, config_data=config_data)

        phase_percent = moon_data.get("phase_percent")
        phase_degrees = moon_data.get("phase_degrees")
        previous_new_moon = moon_data.get("previous_new_moon")
        next_new_moon = moon_data.get("next_new_moon")

        self._phase_percent = phase_percent

        # Round to nearest integer for sensor value (keep as float)
        if phase_degrees is not None:
            degrees_value = round(phase_degrees) % 360
        else:
            degrees_value = None

        self._attr_native_value = degrees_value
        self._attr_available = phase_degrees is not None

        # Calculate next update time at next 1-degree increment
        degree_length_seconds = None
        next_update = None
        next_degree = None
        if phase_degrees is not None and previous_new_moon and next_new_moon:
            # Calculate time per degree
            cycle_duration = (next_new_moon - previous_new_moon).total_seconds()
            degree_length_seconds = cycle_duration / 360.0
            
            # Find next integer degree based on current sensor value
            # Use rounded value to ensure consistency
            next_degree = degrees_value + 1
            if next_degree >= 360:
                # Wrap around: next degree is 0, which is the next new moon
                next_degree = 0
                next_update = next_new_moon + timedelta(seconds=1)
            else:
                # Calculate absolute time when moon will be at next_degree
                # Time from previous_new_moon to next_degree
                time_to_next_degree = next_degree * degree_length_seconds
                next_update = previous_new_moon + timedelta(seconds=time_to_next_degree + 1)
        else:
            # Fallback to hourly if we don't have the data
            next_update = self._get_next_hour(now)
        
        attributes = {
            "next_update": next_update,
            "next_target": next_degree,
        }
        
        if DEBUG_ATTRIBUTES:
            attributes["previous_new_moon"] = previous_new_moon
            attributes["next_new_moon"] = next_new_moon
            attributes["degree_length"] = _format_duration_hhmmss(degree_length_seconds)
        
        self._attr_extra_state_attributes = attributes
        
        return next_update

    def _get_next_hour(self, current_time: datetime) -> datetime:
        next_hour = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return next_hour


class MoonriseSensor(BasePositionSensor):
    """Sensor for moonrise time, updated daily at midnight local time."""
    
    def __init__(self, hass: HomeAssistant, entry_id: str):
        try:
            # Format the name and get unique ID
            name = "Moonrise"
            formatted_name, unique_id = format_sensor_naming(name, entry_id)
            
            # Pass the formatted name and unique_id
            super().__init__(hass=hass, sensor_name=formatted_name, unique_id=unique_id)
            
            # Store entry_id
            self._entry_id = entry_id
            
            # Add device info to link this sensor to the device
            self._attr_device_info = {
                "identifiers": {(DOMAIN, entry_id)},
            }
            
            # Set sensor properties
            self._attr_device_class = "timestamp"
            self._attr_state_class = None  # Timestamp sensors don't need state_class
            
        except Exception as e:
            _LOGGER.error(f"Failed to create MoonriseSensor for entry {entry_id}: {e}", exc_info=True)
            raise
    
    @property
    def device_class(self):
        """Return the device class."""
        return "timestamp"
    
    @property
    def native_unit_of_measurement(self):
        """Return None for timestamp sensors."""
        return None
    
    @property
    def icon(self):
        """Return the icon."""
        return "mdi:weather-night"
    
    async def _async_update_logic(self, now=None) -> datetime | None:
        """Update moonrise time using current date."""
        import zoneinfo
        from datetime import time
        
        # Get local timezone
        local_tz = zoneinfo.ZoneInfo(self.hass.config.time_zone)
        
        # Get current local date
        if now is None:
            now_local = datetime.now(local_tz)
        else:
            now_local = now.astimezone(local_tz)
        
        today = now_local.date()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)
        
        # Calculate moonrise for each date
        config_data = get_config_entry_data(self._entry_id)
        
        yesterday_moonrise, _ = get_moonrise_moonset_for_date(
            self.hass, yesterday, self._entry_id, use_center=False, config_data=config_data
        )
        
        today_moonrise, _ = get_moonrise_moonset_for_date(
            self.hass, today, self._entry_id, use_center=False, config_data=config_data
        )
        
        tomorrow_moonrise, _ = get_moonrise_moonset_for_date(
            self.hass, tomorrow, self._entry_id, use_center=False, config_data=config_data
        )
        
        # Set sensor value to today's moonrise (store as UTC datetime object)
        if today_moonrise:
            # Ensure datetime is timezone-aware and in UTC
            self._attr_native_value = dt_util.as_utc(today_moonrise)
        else:
            self._attr_native_value = None
        
        self._attr_available = today_moonrise is not None
        
        # Calculate next midnight local time
        next_midnight_local = datetime.combine(
            tomorrow, time(0, 0, 0), tzinfo=local_tz
        )
        next_midnight_utc = next_midnight_local.astimezone(timezone.utc)
        
        attributes = {
            "next_update": next_midnight_utc,
            "yesterday": yesterday_moonrise.astimezone(local_tz).isoformat() if yesterday_moonrise else None,
            "today": today_moonrise.astimezone(local_tz).isoformat() if today_moonrise else None,
            "tomorrow": tomorrow_moonrise.astimezone(local_tz).isoformat() if tomorrow_moonrise else None,
        }
        
        self._attr_extra_state_attributes = attributes
        
        return next_midnight_utc


class MoonsetSensor(BasePositionSensor):
    """Sensor for moonset time, updated daily at midnight local time."""
    
    def __init__(self, hass: HomeAssistant, entry_id: str):
        try:
            # Format the name and get unique ID
            name = "Moonset"
            formatted_name, unique_id = format_sensor_naming(name, entry_id)
            
            # Pass the formatted name and unique_id
            super().__init__(hass=hass, sensor_name=formatted_name, unique_id=unique_id)
            
            # Store entry_id
            self._entry_id = entry_id
            
            # Add device info to link this sensor to the device
            self._attr_device_info = {
                "identifiers": {(DOMAIN, entry_id)},
            }
            
            # Set sensor properties
            self._attr_device_class = "timestamp"
            self._attr_state_class = None  # Timestamp sensors don't need state_class
            
        except Exception as e:
            _LOGGER.error(f"Failed to create MoonsetSensor for entry {entry_id}: {e}", exc_info=True)
            raise
    
    @property
    def device_class(self):
        """Return the device class."""
        return "timestamp"
    
    @property
    def native_unit_of_measurement(self):
        """Return None for timestamp sensors."""
        return None
    
    @property
    def icon(self):
        """Return the icon."""
        return "mdi:weather-night"
    
    async def _async_update_logic(self, now=None) -> datetime | None:
        """Update moonset time using current date."""
        import zoneinfo
        from datetime import time
        
        # Get local timezone
        local_tz = zoneinfo.ZoneInfo(self.hass.config.time_zone)
        
        # Get current local date
        if now is None:
            now_local = datetime.now(local_tz)
        else:
            now_local = now.astimezone(local_tz)
        
        today = now_local.date()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)
        
        # Calculate moonset for each date
        config_data = get_config_entry_data(self._entry_id)
        
        _, yesterday_moonset = get_moonrise_moonset_for_date(
            self.hass, yesterday, self._entry_id, use_center=False, config_data=config_data
        )
        
        _, today_moonset = get_moonrise_moonset_for_date(
            self.hass, today, self._entry_id, use_center=False, config_data=config_data
        )
        
        _, tomorrow_moonset = get_moonrise_moonset_for_date(
            self.hass, tomorrow, self._entry_id, use_center=False, config_data=config_data
        )
        
        # Set sensor value to today's moonset (store as UTC datetime object)
        if today_moonset:
            # Ensure datetime is timezone-aware and in UTC
            self._attr_native_value = dt_util.as_utc(today_moonset)
        else:
            self._attr_native_value = None
        
        self._attr_available = today_moonset is not None
        
        # Calculate next midnight local time
        next_midnight_local = datetime.combine(
            tomorrow, time(0, 0, 0), tzinfo=local_tz
        )
        next_midnight_utc = next_midnight_local.astimezone(timezone.utc)
        
        attributes = {
            "next_update": next_midnight_utc,
            "yesterday": yesterday_moonset.astimezone(local_tz).isoformat() if yesterday_moonset else None,
            "today": today_moonset.astimezone(local_tz).isoformat() if today_moonset else None,
            "tomorrow": tomorrow_moonset.astimezone(local_tz).isoformat() if tomorrow_moonset else None,
        }
        
        self._attr_extra_state_attributes = attributes
        
        return next_midnight_utc

