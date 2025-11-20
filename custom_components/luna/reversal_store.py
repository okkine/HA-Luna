"""Persistent storage for azimuth reversal cache."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional
from datetime import timezone

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY_PREFIX = "luna_azimuth_reversals"


class ReversalStore:
    """Manages persistent storage for azimuth reversal cache."""
    
    def __init__(self, hass: HomeAssistant, entry_id: str):
        """Initialize the reversal store."""
        self.hass = hass
        self.entry_id = entry_id
        self._store = Store(
            hass,
            STORAGE_VERSION,
            f"{STORAGE_KEY_PREFIX}_{entry_id}"
        )
    
    async def async_load(self) -> Optional[Dict[str, Any]]:
        """
        Load reversal cache from persistent storage.
        
        Returns:
            Dictionary with cache data or None if not found/invalid
        """
        try:
            data = await self._store.async_load()
            
            if data is None:
                _LOGGER.debug(f"No cached reversal data found for entry {self.entry_id}")
                return None
            
            # Deserialize datetime strings
            if 'last_known_state' in data:
                time_str = data['last_known_state'].get('time')
                if time_str:
                    data['last_known_state']['time'] = datetime.datetime.fromisoformat(time_str)
            
            if 'reversals' in data:
                for reversal in data['reversals']:
                    if 'time' in reversal and isinstance(reversal['time'], str):
                        reversal['time'] = datetime.datetime.fromisoformat(reversal['time'])
            
            _LOGGER.debug(
                f"Loaded reversal cache for entry {self.entry_id}: "
                f"{len(data.get('reversals', []))} reversals"
            )
            
            return data
            
        except Exception as e:
            _LOGGER.error(f"Error loading reversal cache for entry {self.entry_id}: {e}")
            return None
    
    async def async_save(self, data: Dict[str, Any]) -> None:
        """
        Save reversal cache to persistent storage.
        
        Args:
            data: Dictionary with cache data to save
        """
        try:
            # Serialize datetime objects to ISO format strings
            save_data = {}
            
            if 'last_known_state' in data:
                save_data['last_known_state'] = {
                    'time': data['last_known_state']['time'].isoformat(),
                    'direction': data['last_known_state']['direction']
                }
            
            if 'reversals' in data:
                save_data['reversals'] = []
                for reversal in data['reversals']:
                    save_data['reversals'].append({
                        'time': reversal['time'].isoformat(),
                        'azimuth': reversal['azimuth']
                    })
            
            if 'location' in data:
                save_data['location'] = data['location']
            
            await self._store.async_save(save_data)
            
            _LOGGER.debug(
                f"Saved reversal cache for entry {self.entry_id}: "
                f"{len(save_data.get('reversals', []))} reversals"
            )
            
        except Exception as e:
            _LOGGER.error(f"Error saving reversal cache for entry {self.entry_id}: {e}")
    
    async def async_remove(self) -> None:
        """Remove the cached data from storage."""
        try:
            await self._store.async_remove()
            _LOGGER.debug(f"Removed reversal cache for entry {self.entry_id}")
        except Exception as e:
            _LOGGER.error(f"Error removing reversal cache for entry {self.entry_id}: {e}")


async def load_reversal_cache(hass: HomeAssistant, entry_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load reversal cache.
    
    Args:
        hass: Home Assistant instance
        entry_id: Config entry ID
        
    Returns:
        Cache data or None
    """
    store = ReversalStore(hass, entry_id)
    return await store.async_load()


async def save_reversal_cache(hass: HomeAssistant, entry_id: str, data: Dict[str, Any]) -> None:
    """
    Convenience function to save reversal cache.
    
    Args:
        hass: Home Assistant instance
        entry_id: Config entry ID
        data: Cache data to save
    """
    store = ReversalStore(hass, entry_id)
    await store.async_save(data)


async def remove_reversal_cache(hass: HomeAssistant, entry_id: str) -> None:
    """
    Convenience function to remove reversal cache.
    
    Args:
        hass: Home Assistant instance
        entry_id: Config entry ID
    """
    store = ReversalStore(hass, entry_id)
    await store.async_remove()

