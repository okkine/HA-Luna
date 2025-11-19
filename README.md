# Luna - Moon Position & Phase Tracking for Home Assistant

Luna is a comprehensive Home Assistant integration that provides precise tracking of the moon's position, phase, and related astronomical events. Built on the robust PyEphem library, it automatically compensates for atmospheric refraction and intelligently schedules updates to provide accurate data exactly when you need it.

---

## Overview

Luna gives you eight specialized sensors that work together to provide complete moon tracking capabilities. Whether you're interested in the moon's current position in the sky, its phase in the lunar cycle, or when it will rise and set, Luna has you covered.

### What Makes Luna Different?

- **Smart Updates**: Sensors update only when meaningful changes occur, not on arbitrary schedules
- **Atmospheric Compensation**: Automatically adjusts for atmospheric refraction based on your location's elevation and pressure
- **Configurable Precision**: Choose your own step sizes for position tracking to balance accuracy with update frequency
- **Phase-Aware Icons**: Sensor icons automatically reflect the current moon phase in real-time

---

## Sensors

### 🌙 Lunar Elevation

Tracks how high the moon is above the horizon, measured in degrees.

- **Range**: -90° (directly below) to +90° (directly overhead)
- **Updates**: Automatically when the moon reaches each elevation step
- **Default Step**: 0.5° (configurable)

**Attributes:**
- `next_update` - When the sensor will next update
- `next_target` - The elevation value that will trigger the next update

**Example Use**: Trigger automations when the moon rises above your horizon (0°) or reaches a specific elevation.

---

### 🧭 Lunar Azimuth

Tracks the moon's compass direction, measured in degrees.

- **Range**: 0° (North) → 90° (East) → 180° (South) → 270° (West) → 360° (North)
- **Updates**: Automatically when the moon reaches each azimuth step
- **Default Step**: 1.0° (configurable)

**Special Features:**
- Automatically detects and handles azimuth reversals (when the moon changes direction)
- Intelligent caching ensures accurate tracking even in polar regions

**Attributes:**
- `next_update` - When the sensor will next update
- `next_target` - The azimuth value that will trigger the next update

**Example Use**: Know exactly where to look in the sky to see the moon, or trigger lights when the moon is visible through a specific window.

---

### 💡 Lunar Percent Illuminated

Shows how much of the moon's visible surface is currently lit by the sun.

- **Range**: 0% (new moon) to 100% (full moon)
- **Updates**: Every hour, on the hour
- **Icon**: Changes dynamically to match the current moon phase

**Example Use**: Create mood lighting that adjusts based on how bright the moon is, or track the lunar cycle for gardening.

---

### 📊 Lunar Phase Percent

Tracks where the moon is in its monthly cycle as a percentage.

- **Range**: 0-100% (0% = previous new moon, 100% = next new moon)
- **Updates**: Automatically at each 1% increment
- **Icon**: Changes dynamically to match the current moon phase

**Attributes:**
- `next_update` - When the sensor will next update
- `next_target` - The next percentage value (0-100)

**Example Use**: Create precise automations based on the lunar cycle, or display a custom lunar calendar.

---

### 🌓 Lunar Phase

Displays the traditional name of the current moon phase.

- **Phases**: New Moon, Waxing Crescent, First Quarter, Waxing Gibbous, Full Moon, Waning Gibbous, Last Quarter, Waning Crescent
- **Updates**: Automatically when the moon transitions between phases
- **Icon**: Changes dynamically to match the current phase

**Phase Boundaries:**
| Phase | Percentage Range |
|-------|-----------------|
| New Moon | 0-1% or 98-100% |
| Waxing Crescent | 2-22% |
| First Quarter | 23-26% |
| Waxing Gibbous | 27-47% |
| Full Moon | 48-51% |
| Waning Gibbous | 52-72% |
| Last Quarter | 73-76% |
| Waning Crescent | 77-97% |

**Attributes:**
- `next_update` - When the phase will change
- `next_target` - The name of the next phase

**Example Use**: Display the current phase name on your dashboard, or trigger special automations during full moons.

---

### 🔄 Lunar Phase Degrees

Tracks the moon's position in its cycle using degrees instead of percentages.

- **Range**: 0-359° (0° = previous new moon, 360° = next new moon)
- **Updates**: Automatically at each 1° increment
- **Icon**: Changes dynamically to match the current moon phase

**Attributes:**
- `next_update` - When the sensor will next update
- `next_target` - The next degree value (0-359)

**Example Use**: For users who prefer working with degrees, or for creating precise astronomical calculations.

**Advanced Use - Visual Moon Phase Display**: Create a picture-elements card that displays the exact current moon phase image. Using the degree value (0-359), you can dynamically show the corresponding moon phase image from [Jay Tanner's Lunar Phase Set](https://commons.wikimedia.org/wiki/User:JayTanner/lunar-near-side-phase-set), which provides 361 high-resolution images at 1-degree intervals (0° = New Moon, 90° = First Quarter, 180° = Full Moon, 270° = Last Quarter). This creates a live, pixel-perfect representation of the moon's current appearance.

---

### 🌅 Moonrise

Shows when the moon will rise above the horizon today.

- **Updates**: Daily at midnight (local time)
- **Format**: Automatically formatted by Home Assistant based on your preferences
- **Calculation**: Uses the moon's edge (not center) for practical visibility

**Attributes:**
- `yesterday` - Yesterday's moonrise time
- `today` - Today's moonrise time
- `tomorrow` - Tomorrow's moonrise time
- `next_update` - When the sensor will refresh (next midnight)

**Note**: Some days may not have a moonrise (when the moon is circumpolar or doesn't rise that day).

---

### 🌇 Moonset

Shows when the moon will set below the horizon today.

- **Updates**: Daily at midnight (local time)
- **Format**: Automatically formatted by Home Assistant based on your preferences
- **Calculation**: Uses the moon's edge (not center) for practical visibility

**Attributes:**
- `yesterday` - Yesterday's moonset time
- `today` - Today's moonset time
- `tomorrow` - Tomorrow's moonset time
- `next_update` - When the sensor will refresh (next midnight)

**Note**: Some days may not have a moonset (when the moon is circumpolar or doesn't set that day).

---

## Atmospheric Compensation

Luna automatically accounts for how Earth's atmosphere bends light from the moon, making it appear higher in the sky than its true astronomical position. This effect is most noticeable near the horizon.

### Apparent vs. True Position

**By default, Luna shows the moon's *apparent position*** - where it appears to be in the sky when you look at it. This is what you'll actually see with your eyes or through a telescope, accounting for atmospheric refraction.

If you prefer the **true astronomical position** (the moon's geometric position without atmospheric effects, like most other sun/moon position sensors), you can set the atmospheric pressure to `0` during configuration. This disables refraction correction entirely.

**How It Works:**
- Uses your location's elevation to calculate typical atmospheric pressure
- Applies standard astronomical refraction models to calculate apparent position
- Fully configurable - set pressure to `0` for true position, or customize for your local conditions

**Default Settings:**
- Pressure: 1013.25 mbar (standard sea level pressure)
- Automatically adjusted based on your configured elevation
- **Result**: Shows apparent position (what you see in the sky)

**For True Astronomical Position:**
- Set pressure to `0` mbar during configuration
- **Result**: Shows geometric position (no atmospheric correction)

Most users should use the default settings, as the apparent position is more useful for practical observations and photography.

---

## Configuration

### Initial Setup

1. Install Luna through HACS or manually
2. Go to **Settings** → **Devices & Services** → **Add Integration**
3. Search for "Luna" and select it
4. Choose your location (use Home Assistant's location or specify a custom one)
5. Optionally adjust elevation and atmospheric pressure settings

### Customizing Step Values

You can adjust how frequently the position sensors update:

- **Elevation Step**: How many degrees the moon must move vertically before updating (minimum 0.1)
  - Smaller values (e.g., 0.1°) = more frequent updates, higher precision 
  - Larger values (e.g., 2.0°) = fewer updates, lower resource usage
  - Default: 0.5°

- **Azimuth Step**: How many degrees the moon must move horizontally before updating
  - Smaller values (e.g., 0.5°) = more frequent updates, higher precision (can be problematic at tropical latitudes)
  - Larger values (e.g., 5.0°) = fewer updates, lower resource usage
  - Default: 1.0°

**Recommendation**: Start with the defaults. Only decrease step values if you need higher precision for specific automations.

---

## Technical Details

### Calculation Engine
- **Library**: PyEphem 4.1.4+ (industry-standard astronomical calculations)
- **Precision**: Sub-degree accuracy for all position data
- **Time Handling**: All calculations respect your Home Assistant timezone

### Coordinate Systems
- **Azimuth**: 0° = North, increasing clockwise (East = 90°, South = 180°, West = 270°)
- **Elevation**: 0° = horizon, 90° = directly overhead (zenith), -90° = directly below (nadir)

### Update Strategy
Luna uses event-driven updates rather than polling:
- Position sensors calculate when the next threshold will be reached
- Phase sensors update at natural boundaries
- Time sensors update once per day
- No unnecessary calculations or updates

This approach minimizes CPU usage while ensuring data is always current.

---

## Installation

### Via HACS (Recommended)

1. Open HACS in Home Assistant
2. Click the three dots in the top right and select "Custom repositories"
3. Add `https://github.com/Okkine/HA-Luna` as an Integration
4. Click "Install"
5. Restart Home Assistant
6. Add the Luna integration through the UI

### Manual Installation

1. Download the latest release from GitHub
2. Copy the `custom_components/luna` folder to your Home Assistant `custom_components` directory
3. Restart Home Assistant
4. Add the Luna integration through the UI

---

## Requirements

- **Home Assistant**: 2023.8.0 or later
- **Python Packages**: Automatically installed
  - `ephem` >= 4.1.4
  - `python-slugify`
  - `ambiance`

---

## Support & Contributing

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/Okkine/HA-Luna/issues)
- **Discussions**: Ask questions or share ideas on [GitHub Discussions](https://github.com/Okkine/HA-Luna/discussions)
- **Contributing**: Pull requests are welcome! Please read the contributing guidelines first.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Built with [PyEphem](https://rhodesmill.org/pyephem/) for astronomical calculations
- Lunar phase images by [Jay Tanner](https://commons.wikimedia.org/wiki/User:JayTanner/lunar-near-side-phase-set) (Wikimedia Commons, used under CC0 License) - used for integration logo and available for visual phase displays
- Inspired by the Home Assistant community's need for accurate lunar tracking
- Thanks to all contributors and users who provide feedback and suggestions
