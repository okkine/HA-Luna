# Luna - Moon Position & Phase Integration

A comprehensive Home Assistant integration for tracking the moon's position, phase, and events with high precision and automatic atmospheric corrections.

## Features

Luna provides eight specialized sensors that track various aspects of the moon's position and phase, with intelligent update scheduling and atmospheric distortion compensation.

## Sensors

### Lunar Elevation
Tracks the moon's elevation angle above the horizon in degrees (0° = horizon, 90° = zenith).

- **Update Frequency**: Updates automatically when the moon reaches the next elevation step
- **Step Size**: Configurable (default: 0.5°)
- **Attributes**:
  - `next_update`: Next scheduled update time
  - `next_target`: Next elevation target in degrees
  - `percent_illuminated`: Current moon illumination percentage

### Lunar Azimuth
Tracks the moon's azimuth angle in degrees (0° = North, 90° = East, 180° = South, 270° = West).

- **Update Frequency**: Updates automatically when the moon reaches the next azimuth step
- **Step Size**: Configurable (default: 1.0°)
- **Special Features**:
  - Automatic detection and handling of azimuth reversals (when the moon's azimuth direction changes)
  - Intelligent reversal caching for accurate tracking near the poles
- **Attributes**:
  - `next_update`: Next scheduled update time
  - `next_target`: Next azimuth target in degrees
  - `percent_illuminated`: Current moon illumination percentage

### Lunar Percent Illuminated
Displays the percentage of the moon's surface illuminated by the sun (0-100%).

- **Update Frequency**: Updates hourly on the hour
- **Icon**: Dynamically changes based on current moon phase

### Lunar Phase Percent
Tracks the moon's position in the lunar cycle as a percentage (0-100%, where 0% = previous new moon, 100% = next new moon).

- **Update Frequency**: Updates automatically at each 1% increment
- **Attributes**:
  - `next_update`: Next scheduled update time
  - `next_target`: Next integer percent value (0-100)

### Lunar Phase
Displays the current moon phase name (e.g., "New Moon", "Waxing Crescent", "First Quarter", etc.).

- **Update Frequency**: Updates automatically at phase threshold boundaries
- **Phase Thresholds**:
  - New Moon: ≤ 1% or ≥ 98%
  - Waxing Crescent: 2-22%
  - First Quarter: 23-26%
  - Waxing Gibbous: 27-47%
  - Full Moon: 48-51%
  - Waning Gibbous: 52-72%
  - Last Quarter: 73-76%
  - Waning Crescent: 77-97%
- **Attributes**:
  - `next_update`: Next scheduled update time
  - `next_target`: Next phase name

### Lunar Phase Degrees
Tracks the moon's position in the lunar cycle in degrees (0-359°).

- **Update Frequency**: Updates automatically at each 1° increment
- **Attributes**:
  - `next_update`: Next scheduled update time
  - `next_target`: Next integer degree value (0-359)

### Moonrise
Displays the moonrise time for today.

- **Update Frequency**: Updates daily at midnight local time
- **Device Class**: Timestamp (formatted automatically by Home Assistant)
- **Attributes**:
  - `yesterday`: Yesterday's moonrise time
  - `today`: Today's moonrise time
  - `tomorrow`: Tomorrow's moonrise time
  - `next_update`: Next scheduled update time

### Moonset
Displays the moonset time for today.

- **Update Frequency**: Updates daily at midnight local time
- **Device Class**: Timestamp (formatted automatically by Home Assistant)
- **Attributes**:
  - `yesterday`: Yesterday's moonset time
  - `today`: Today's moonset time
  - `tomorrow`: Tomorrow's moonset time
  - `next_update`: Next scheduled update time

## Atmospheric Distortion Compensation

All position calculations automatically account for atmospheric refraction using configurable atmospheric pressure settings:

- **Default Pressure**: 1013.25 mbar (standard atmospheric pressure at sea level)
- **Configurable**: Can be set manually or automatically calculated from elevation
- **Effect**: Corrects for atmospheric bending of light, providing more accurate elevation and azimuth values

The integration uses the configured pressure value in all astronomical calculations via the PyEphem library, ensuring accurate moon position tracking regardless of your location's elevation or atmospheric conditions.

## Configuration

### Step Values

Both elevation and azimuth sensors support configurable step sizes:

- **Elevation Step**: Default 0.5° (configurable)
- **Azimuth Step**: Default 1.0° (configurable)

Smaller step values provide more frequent updates and higher precision, while larger steps reduce update frequency and computational load.

### Update Scheduling

Sensors use intelligent update scheduling:

- **Position Sensors** (Elevation/Azimuth): Update only when the moon reaches the next step threshold
- **Phase Sensors**: Update at natural phase boundaries or increments
- **Time Sensors** (Moonrise/Moonset): Update daily at midnight local time
- **Percent Illuminated**: Updates hourly on the hour

This approach minimizes unnecessary updates while ensuring data accuracy.

## Technical Details

- **Calculation Library**: PyEphem (astronomical calculations)
- **Coordinate System**: 
  - Azimuth: 0° = North, clockwise positive
  - Elevation: 0° = horizon, 90° = zenith
- **Time Zone Handling**: All calculations respect your Home Assistant timezone configuration
- **Precision**: Sub-degree accuracy with configurable step sizes

## Installation

Install via HACS (Home Assistant Community Store) or manually by copying the `luna` folder to your `custom_components` directory.

## Requirements

- Home Assistant 2024.1.0 or later
- PyEphem library (automatically installed)

## Support

For issues, feature requests, or contributions, please visit the [GitHub repository](https://github.com/yourusername/HA-Luna).
