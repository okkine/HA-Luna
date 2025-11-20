"""Azimuth reversal cache manager."""

from __future__ import annotations

import datetime
import logging
import math
import zoneinfo
from datetime import timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import ephem

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    AZIMUTH_REVERSAL_CACHE_LENGTH,
    AZIMUTH_REVERSAL_MAX_SEARCH_DAYS,
    AZIMUTH_REVERSAL_SEARCH_MAX_ITERATIONS,
    AZIMUTH_DEGREE_TOLERANCE,
    TROPICAL_LATITUDE_THRESHOLD,
)
from .reversal_store import load_reversal_cache, save_reversal_cache, remove_reversal_cache
from .utils import get_moon_position, calculate_azimuth_derivative
from .config_store import get_config_entry_data

_LOGGER = logging.getLogger(__name__)


class ReversalCacheManager:
    """Manages azimuth reversal cache with persistent storage."""
    
    def __init__(self, hass: HomeAssistant):
        """Initialize the reversal cache manager."""
        self.hass = hass
        self._maintenance_timers = {}  # Dict[entry_id, timer]
        self._daily_check_timers = {}  # Dict[entry_id, timer]
        self._weekly_check_timers = {}  # Dict[entry_id, timer]
        self._initialized_entries = set()
    
    async def initialize_entry(self, entry_id: str) -> None:
        """
        Initialize reversal cache for an entry.
        
        Args:
            entry_id: Config entry ID
        """
        if entry_id in self._initialized_entries:
            _LOGGER.debug(f"Entry {entry_id} already initialized")
            return
        
        try:
            # Try to load existing cache
            cache = await load_reversal_cache(self.hass, entry_id)
            
            if cache is None:
                # No cache exists, initialize from scratch
                _LOGGER.info(f"Initializing new reversal cache for entry {entry_id}")
                cache = await self._initialize_from_lunar_transit(entry_id)
            else:
                # Cache exists, validate and clean up
                _LOGGER.info(f"Loaded existing reversal cache for entry {entry_id}")
                cache = await self._validate_and_cleanup_cache(entry_id, cache)
            
            # Schedule maintenance and checks
            await self._schedule_next_maintenance(entry_id)
            await self._schedule_recalculation_check(entry_id)
            
            self._initialized_entries.add(entry_id)
            
            _LOGGER.info(
                f"Reversal cache initialized for {entry_id}: "
                f"{len(cache['reversals'])} future reversals"
            )
            
        except Exception as e:
            _LOGGER.error(f"Error initializing reversal cache for entry {entry_id}: {e}", exc_info=True)
    
    async def remove_entry(self, entry_id: str) -> None:
        """
        Remove reversal cache for an entry.
        
        Args:
            entry_id: Config entry ID
        """
        # Cancel timers
        if entry_id in self._maintenance_timers:
            self._maintenance_timers[entry_id].cancel()
            del self._maintenance_timers[entry_id]
        
        if entry_id in self._daily_check_timers:
            self._daily_check_timers[entry_id].cancel()
            del self._daily_check_timers[entry_id]
        
        if entry_id in self._weekly_check_timers:
            self._weekly_check_timers[entry_id].cancel()
            del self._weekly_check_timers[entry_id]
        
        # Remove from initialized set
        self._initialized_entries.discard(entry_id)
        
        # Remove persistent storage
        await remove_reversal_cache(self.hass, entry_id)
        
        _LOGGER.info(f"Removed reversal cache for entry {entry_id}")
    
    async def get_reversals(self, entry_id: str, current_time: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """
        Get reversal cache for an entry.
        
        Args:
            entry_id: Config entry ID
            current_time: Current time (defaults to now)
            
        Returns:
            Cache dictionary with last_known_state and reversals
        """
        if current_time is None:
            current_time = dt_util.now()
        
        # Ensure initialized
        if entry_id not in self._initialized_entries:
            await self.initialize_entry(entry_id)
        
        # Load cache
        cache = await load_reversal_cache(self.hass, entry_id)
        
        if cache is None:
            # Something went wrong, reinitialize
            _LOGGER.warning(f"Cache missing for initialized entry {entry_id}, reinitializing")
            cache = await self._initialize_from_lunar_transit(entry_id)
        
        return cache
    
    def get_current_direction(self, cache: Dict[str, Any], current_time: datetime.datetime) -> int:
        """
        Calculate current azimuth direction from cache.
        
        Args:
            cache: Cache dictionary
            current_time: Current time
            
        Returns:
            Current direction (1 or -1)
        """
        direction = cache['last_known_state']['direction']
        last_known_time = cache['last_known_state']['time']
        
        # Count reversals between last_known_time and current_time
        for reversal in cache['reversals']:
            if last_known_time < reversal['time'] <= current_time:
                direction *= -1
        
        return direction
    
    async def _initialize_from_lunar_transit(self, entry_id: str) -> Dict[str, Any]:
        """
        Initialize cache from previous lunar transit.
        
        Args:
            entry_id: Config entry ID
            
        Returns:
            Initialized cache dictionary
        """
        config_data = get_config_entry_data(entry_id)
        now = dt_util.now()
        now_utc = now.astimezone(timezone.utc)
        
        # Get previous lunar transit
        observer = ephem.Observer()
        observer.lat = str(config_data.get('latitude', self.hass.config.latitude))
        observer.lon = str(config_data.get('longitude', self.hass.config.longitude))
        observer.elevation = config_data.get('elevation', self.hass.config.elevation)
        observer.date = now_utc
        
        moon = ephem.Moon()
        prev_lunar_transit = observer.previous_transit(moon)
        prev_lunar_transit_dt = prev_lunar_transit.datetime().replace(tzinfo=timezone.utc)
        
        _LOGGER.debug(f"Initializing from previous lunar transit: {prev_lunar_transit_dt}")
        
        # Sample direction at previous lunar transit
        transit_minus_10 = prev_lunar_transit_dt - timedelta(minutes=10)
        transit_plus_10 = prev_lunar_transit_dt + timedelta(minutes=10)
        
        az_before = get_moon_position(self.hass, transit_minus_10, entry_id, config_data=config_data)['azimuth']
        az_after = get_moon_position(self.hass, transit_plus_10, entry_id, config_data=config_data)['azimuth']
        
        derivative = calculate_azimuth_derivative(az_before, az_after)
        direction_at_transit = 1 if derivative > 0 else -1
        
        _LOGGER.debug(f"Direction at lunar transit: {direction_at_transit}")
        
        # Scan from previous transit forward
        search_end = now_utc + timedelta(days=AZIMUTH_REVERSAL_MAX_SEARCH_DAYS)
        all_reversals = await self._scan_for_reversals(
            entry_id,
            start_time=prev_lunar_transit_dt,
            end_time=search_end,
            start_direction=direction_at_transit,
            config_data=config_data
        )
        
        # Split into passed and future
        passed = [r for r in all_reversals if r['time'] <= now_utc]
        future = [r for r in all_reversals if r['time'] > now_utc]
        
        _LOGGER.debug(f"Found {len(passed)} passed and {len(future)} future reversals")
        
        # Calculate current direction
        current_direction = direction_at_transit
        for _ in passed:
            current_direction *= -1
        
        # Set last_known_state
        if passed:
            last_known_time = passed[-1]['time']
            last_known_direction = current_direction
        else:
            last_known_time = prev_lunar_transit_dt
            last_known_direction = direction_at_transit
        
        # Keep up to AZIMUTH_REVERSAL_CACHE_LENGTH future reversals
        cache = {
            'last_known_state': {
                'time': last_known_time,
                'direction': last_known_direction
            },
            'reversals': future[:AZIMUTH_REVERSAL_CACHE_LENGTH],
            'location': {
                'latitude': config_data.get('latitude', self.hass.config.latitude),
                'longitude': config_data.get('longitude', self.hass.config.longitude)
            }
        }
        
        await save_reversal_cache(self.hass, entry_id, cache)
        
        return cache
    
    async def _validate_and_cleanup_cache(
        self, 
        entry_id: str, 
        cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and clean up loaded cache.
        
        Args:
            entry_id: Config entry ID
            cache: Loaded cache dictionary
            
        Returns:
            Validated and cleaned cache
        """
        config_data = get_config_entry_data(entry_id)
        now_utc = dt_util.now().astimezone(timezone.utc)
        
        # Check if location changed
        cached_lat = cache.get('location', {}).get('latitude')
        cached_lon = cache.get('location', {}).get('longitude')
        current_lat = config_data.get('latitude', self.hass.config.latitude)
        current_lon = config_data.get('longitude', self.hass.config.longitude)
        
        if cached_lat != current_lat or cached_lon != current_lon:
            _LOGGER.info(f"Location changed for {entry_id}, reinitializing cache")
            return await self._initialize_from_lunar_transit(entry_id)
        
        # Remove past reversals and update last_known_state
        passed_reversals = [r for r in cache['reversals'] if r['time'] <= now_utc]
        
        if passed_reversals:
            _LOGGER.debug(f"Removing {len(passed_reversals)} passed reversals")
            
            # Update last_known_state
            direction = cache['last_known_state']['direction']
            for _ in passed_reversals:
                direction *= -1
            
            cache['last_known_state'] = {
                'time': passed_reversals[-1]['time'],
                'direction': direction
            }
            
            # Remove passed reversals
            cache['reversals'] = [r for r in cache['reversals'] if r['time'] > now_utc]
        
        # Ensure we have enough future reversals
        if len(cache['reversals']) < AZIMUTH_REVERSAL_CACHE_LENGTH:
            _LOGGER.debug(f"Need more reversals, calculating...")
            cache = await self._refill_reversals(entry_id, cache, config_data)
        
        await save_reversal_cache(self.hass, entry_id, cache)
        
        return cache
    
    async def _refill_reversals(
        self,
        entry_id: str,
        cache: Dict[str, Any],
        config_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate additional reversals to reach AZIMUTH_REVERSAL_CACHE_LENGTH.
        
        Args:
            entry_id: Config entry ID
            cache: Current cache
            config_data: Config data
            
        Returns:
            Updated cache
        """
        # Determine starting point for new search
        if cache['reversals']:
            search_start = cache['reversals'][-1]['time']
            start_direction = self.get_current_direction(cache, search_start)
        else:
            search_start = cache['last_known_state']['time']
            start_direction = cache['last_known_state']['direction']
        
        search_end = search_start + timedelta(days=AZIMUTH_REVERSAL_MAX_SEARCH_DAYS)
        
        # Scan for more reversals
        new_reversals = await self._scan_for_reversals(
            entry_id,
            start_time=search_start,
            end_time=search_end,
            start_direction=start_direction,
            config_data=config_data
        )
        
        # Add new reversals to cache
        cache['reversals'].extend(new_reversals)
        cache['reversals'] = cache['reversals'][:AZIMUTH_REVERSAL_CACHE_LENGTH]
        
        return cache
    
    async def _scan_for_reversals(
        self,
        entry_id: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        start_direction: int,
        config_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Scan for azimuth reversals in time window.
        
        Args:
            entry_id: Config entry ID
            start_time: Start of scan window
            end_time: End of scan window
            start_direction: Known direction at start_time
            config_data: Config data
            
        Returns:
            List of reversal dictionaries
        """
        reversals = []
        current_direction = start_direction
        search_time = start_time + timedelta(minutes=5)
        
        while search_time < end_time:
            # Sample azimuth change over 10-minute window
            az_before = get_moon_position(self.hass, search_time, entry_id, config_data=config_data)['azimuth']
            az_after = get_moon_position(
                self.hass,
                search_time + timedelta(minutes=10),
                entry_id,
                config_data=config_data
            )['azimuth']
            
            derivative = calculate_azimuth_derivative(az_before, az_after)
            observed_direction = 1 if derivative > 0 else -1
            
            # Check for direction change
            if observed_direction != current_direction:
                # Binary search to find exact reversal
                try:
                    reversal_time, reversal_azimuth = await self._binary_search_reversal(
                        search_time - timedelta(minutes=5),
                        search_time + timedelta(minutes=10),
                        entry_id,
                        config_data
                    )
                    
                    reversals.append({
                        'time': reversal_time,
                        'azimuth': reversal_azimuth
                    })
                    
                    current_direction *= -1
                    
                except Exception as e:
                    _LOGGER.warning(f"Error finding reversal: {e}")
            
            search_time += timedelta(minutes=5)
        
        if len(reversals) < AZIMUTH_REVERSAL_CACHE_LENGTH:
            _LOGGER.info(
                f"Only found {len(reversals)} reversals within "
                f"{AZIMUTH_REVERSAL_MAX_SEARCH_DAYS} days for entry {entry_id}"
            )
        
        return reversals
    
    async def _binary_search_reversal(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        entry_id: str,
        config_data: Dict[str, Any]
    ) -> Tuple[datetime.datetime, float]:
        """
        Binary search to find exact reversal point.
        
        Args:
            start_time: Start of search window
            end_time: End of search window
            entry_id: Config entry ID
            config_data: Config data
            
        Returns:
            Tuple of (reversal_time, reversal_azimuth)
        """
        left, right = start_time, end_time
        iterations = 0
        
        while iterations < AZIMUTH_REVERSAL_SEARCH_MAX_ITERATIONS:
            # Check azimuth difference across current window
            left_azimuth = get_moon_position(self.hass, left, entry_id, config_data=config_data)['azimuth']
            right_azimuth = get_moon_position(self.hass, right, entry_id, config_data=config_data)['azimuth']
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
            left_az = get_moon_position(self.hass, left, entry_id, config_data=config_data)['azimuth']
            left_mid_az = get_moon_position(self.hass, left_mid, entry_id, config_data=config_data)['azimuth']
            right_mid_az = get_moon_position(self.hass, right_mid, entry_id, config_data=config_data)['azimuth']
            right_az = get_moon_position(self.hass, right, entry_id, config_data=config_data)['azimuth']
            
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
        final_azimuth = get_moon_position(self.hass, final_time, entry_id, config_data=config_data)['azimuth']
        
        return final_time, final_azimuth
    
    async def _schedule_next_maintenance(self, entry_id: str) -> None:
        """Schedule maintenance for after next reversal."""
        # Cancel existing timer
        if entry_id in self._maintenance_timers:
            self._maintenance_timers[entry_id].cancel()
        
        try:
            cache = await load_reversal_cache(self.hass, entry_id)
            
            if cache and cache['reversals']:
                # Schedule for 100ms after first reversal
                next_reversal_time = cache['reversals'][0]['time']
                maintenance_time = next_reversal_time + timedelta(milliseconds=100)
                
                now = dt_util.now().astimezone(timezone.utc)
                delay = (maintenance_time - now).total_seconds()
                
                if delay > 0:
                    self._maintenance_timers[entry_id] = self.hass.loop.call_later(
                        delay,
                        lambda: self.hass.async_create_task(
                            self._perform_maintenance(entry_id)
                        )
                    )
                    _LOGGER.debug(
                        f"Scheduled maintenance for {entry_id} in {delay/3600:.2f} hours "
                        f"at {maintenance_time}"
                    )
                else:
                    # Reversal already passed, run maintenance immediately
                    await self._perform_maintenance(entry_id)
        
        except Exception as e:
            _LOGGER.error(f"Error scheduling maintenance for {entry_id}: {e}")
    
    async def _perform_maintenance(self, entry_id: str) -> None:
        """Perform scheduled maintenance."""
        try:
            _LOGGER.debug(f"Performing maintenance for {entry_id}")
            
            cache = await load_reversal_cache(self.hass, entry_id)
            if not cache:
                _LOGGER.warning(f"No cache found during maintenance for {entry_id}")
                return
            
            config_data = get_config_entry_data(entry_id)
            now_utc = dt_util.now().astimezone(timezone.utc)
            
            # Check if any reversals have passed
            passed_reversals = [r for r in cache['reversals'] if r['time'] <= now_utc]
            
            if passed_reversals:
                _LOGGER.debug(f"Updating last_known_state after {len(passed_reversals)} passed reversals")
                
                # Update last_known_state
                direction = cache['last_known_state']['direction']
                for _ in passed_reversals:
                    direction *= -1
                
                cache['last_known_state'] = {
                    'time': passed_reversals[-1]['time'],
                    'direction': direction
                }
                
                # Remove passed reversals
                cache['reversals'] = [r for r in cache['reversals'] if r['time'] > now_utc]
            
            # Ensure we have enough future reversals
            if len(cache['reversals']) < AZIMUTH_REVERSAL_CACHE_LENGTH:
                cache = await self._refill_reversals(entry_id, cache, config_data)
            
            await save_reversal_cache(self.hass, entry_id, cache)
            
            # Reschedule for next reversal
            await self._schedule_next_maintenance(entry_id)
            
        except Exception as e:
            _LOGGER.error(f"Error during maintenance for {entry_id}: {e}", exc_info=True)
    
    async def _schedule_recalculation_check(self, entry_id: str) -> None:
        """Schedule daily (tropical) or weekly (non-tropical) recalculation check."""
        config_data = get_config_entry_data(entry_id)
        latitude = config_data.get('latitude', self.hass.config.latitude)
        
        if self._is_tropical_location(latitude):
            await self._schedule_daily_check(entry_id)
        else:
            await self._schedule_weekly_check(entry_id)
    
    def _is_tropical_location(self, latitude: float) -> bool:
        """Check if location is within tropics."""
        return abs(latitude) <= TROPICAL_LATITUDE_THRESHOLD
    
    async def _schedule_daily_check(self, entry_id: str) -> None:
        """Schedule daily midnight check for tropical locations."""
        # Cancel existing timer
        if entry_id in self._daily_check_timers:
            self._daily_check_timers[entry_id].cancel()
        
        local_tz = zoneinfo.ZoneInfo(self.hass.config.time_zone)
        now = dt_util.now(local_tz)
        
        # Next midnight
        tomorrow_midnight = datetime.datetime.combine(
            now.date() + timedelta(days=1),
            datetime.time(0, 0, 0),
            tzinfo=local_tz
        )
        
        delay = (tomorrow_midnight - now).total_seconds()
        
        self._daily_check_timers[entry_id] = self.hass.loop.call_later(
            delay,
            lambda: self.hass.async_create_task(
                self._perform_daily_check(entry_id)
            )
        )
        
        _LOGGER.debug(f"Scheduled daily check for {entry_id} in {delay/3600:.1f} hours")
    
    async def _perform_daily_check(self, entry_id: str) -> None:
        """Perform daily check for tropical location."""
        try:
            _LOGGER.debug(f"Performing daily check for {entry_id}")
            
            # Just ensure cache is still valid and has enough reversals
            cache = await load_reversal_cache(self.hass, entry_id)
            if cache:
                config_data = get_config_entry_data(entry_id)
                cache = await self._validate_and_cleanup_cache(entry_id, cache)
            
            # Reschedule for tomorrow
            await self._schedule_daily_check(entry_id)
            
        except Exception as e:
            _LOGGER.error(f"Error during daily check for {entry_id}: {e}")
    
    async def _schedule_weekly_check(self, entry_id: str) -> None:
        """Schedule weekly validation for non-tropical locations."""
        # Cancel existing timer
        if entry_id in self._weekly_check_timers:
            self._weekly_check_timers[entry_id].cancel()
        
        delay = 7 * 24 * 3600  # 7 days in seconds
        
        self._weekly_check_timers[entry_id] = self.hass.loop.call_later(
            delay,
            lambda: self.hass.async_create_task(
                self._perform_weekly_check(entry_id)
            )
        )
        
        _LOGGER.debug(f"Scheduled weekly check for {entry_id} in 7 days")
    
    async def _perform_weekly_check(self, entry_id: str) -> None:
        """Perform weekly check for non-tropical location."""
        try:
            _LOGGER.debug(f"Performing weekly check for {entry_id}")
            
            # Validate cache
            cache = await load_reversal_cache(self.hass, entry_id)
            if cache:
                config_data = get_config_entry_data(entry_id)
                cache = await self._validate_and_cleanup_cache(entry_id, cache)
            
            # Reschedule for next week
            await self._schedule_weekly_check(entry_id)
            
        except Exception as e:
            _LOGGER.error(f"Error during weekly check for {entry_id}: {e}")


# Global cache manager instance
_cache_manager_instance: Optional[ReversalCacheManager] = None


def get_reversal_cache_manager(hass: HomeAssistant) -> ReversalCacheManager:
    """Get the global reversal cache manager instance."""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = ReversalCacheManager(hass)
    return _cache_manager_instance

