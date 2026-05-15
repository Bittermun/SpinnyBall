"""
SPICE wrapper for high-fidelity ephemeris and frame transformations.

This module provides a lightweight interface to NAIF SPICE kernels via spiceypy,
enabling access to precise body positions, velocities, and time conversions.

Features:
- Automatic kernel loading and caching
- Body state queries (Earth, Moon, Sun)
- Frame transformations (ECI <-> other frames)
- Time system conversions (UTC, TDB, JD)
- Kernel file management and fallback

Requirements:
- spiceypy (pip install spiceypy)
- NAIF SPICE kernels (de430.bsp, pck00010.tpc, etc.)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

# Try to import spiceypy
try:
    import spiceypy as spice
    SPICEYPY_AVAILABLE = True
except ImportError:
    SPICEYPY_AVAILABLE = False
    spice = None


@dataclass
class SPICEState:
    """
    Ephemeris state from SPICE query.
    
    Attributes:
        position: Position vector [x, y, z] (km)
        velocity: Velocity vector [vx, vy, vz] (km/s)
        time_jd: Time as Julian Date
        frame: Reference frame (default 'J2000')
        body_name: Body name queried
    """
    position: np.ndarray
    velocity: np.ndarray
    time_jd: float
    frame: str = 'J2000'
    body_name: str = 'MOON'
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)


class SPICEWrapper:
    """
    Lightweight SPICE kernel wrapper for cislunar propagation.
    
    Provides:
    - Body state vectors (Earth, Moon, Sun)
    - Frame transformations
    - Time system conversions
    - Kernel caching
    
    Usage:
        spw = SPICEWrapper(kernel_dir='/path/to/kernels')
        state = spw.get_body_state('MOON', time_jd=2460000.0)
        print(f"Moon position: {state.position} km")
    """
    
    # NAIF body IDs
    BODY_IDS = {
        'EARTH': 399,
        'MOON': 301,
        'SUN': 10,
        'BARYCENTER': 0,
        'EMB': 3,  # Earth-Moon barycenter
    }
    
    def __init__(
        self,
        kernel_dir: Optional[Path] = None,
        auto_load_kernels: bool = True,
        verbose: bool = False
    ):
        """
        Initialize SPICE wrapper.
        
        Args:
            kernel_dir: Directory containing SPICE kernel files (or None for default)
            auto_load_kernels: Automatically load common kernels on init
            verbose: Print kernel loading messages
        
        Raises:
            ImportError: If spiceypy is not installed
        """
        if not SPICEYPY_AVAILABLE:
            raise ImportError(
                "spiceypy not installed. Install with: pip install spiceypy\n"
                "Download NAIF kernels from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
            )
        
        self.kernel_dir = kernel_dir or self._get_default_kernel_dir()
        self.verbose = verbose
        self._kernels_loaded = False
        self._kernel_cache = {}
        
        if auto_load_kernels:
            self.load_kernels()
    
    @staticmethod
    def _get_default_kernel_dir() -> Path:
        """
        Get default kernel directory.
        
        Looks for kernels in:
        1. ./kernels/
        2. ../kernels/
        3. ./data/kernels/
        """
        candidates = [
            Path('./kernels'),
            Path('../kernels'),
            Path('./data/kernels'),
            Path(spice.furnsh.__module__).parent / 'data' if SPICEYPY_AVAILABLE else None,
        ]
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        # Default to current directory if none found
        return Path('./kernels')
    
    def load_kernels(self, kernel_dir: Optional[Path] = None) -> None:
        """
        Load SPICE kernel files from directory.
        
        Standard kernels loaded:
        - de430.bsp (planetary ephemeris)
        - pck00010.tpc (planet/satellite orientation)
        - naif0012.tls (leap seconds)
        
        Args:
            kernel_dir: Override kernel directory for this load
        """
        if kernel_dir is None:
            kernel_dir = self.kernel_dir
        
        if not kernel_dir.exists():
            if self.verbose:
                print(f"Warning: kernel directory not found: {kernel_dir}")
            return
        
        # Load common kernels in order
        kernel_names = [
            'de430.bsp',           # Planetary ephemeris
            'pck00010.tpc',        # Orientation constants
            'naif0012.tls',        # Leap seconds
            'earth_latest_high_prec.bpc',  # Earth high-precision orientation (optional)
        ]
        
        loaded_count = 0
        for kernel_name in kernel_names:
            kernel_path = kernel_dir / kernel_name
            if kernel_path.exists():
                try:
                    spice.furnsh(str(kernel_path))
                    self._kernel_cache[kernel_name] = str(kernel_path)
                    if self.verbose:
                        print(f"Loaded kernel: {kernel_name}")
                    loaded_count += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to load {kernel_name}: {e}")
        
        self._kernels_loaded = (loaded_count > 0)
        if self.verbose:
            print(f"Loaded {loaded_count} SPICE kernels")
    
    def get_body_state(
        self,
        body_name: str,
        time_jd: float,
        observer: str = 'EARTH',
        frame: str = 'J2000',
        aberration: str = 'NONE'
    ) -> SPICEState:
        """
        Query body state from SPICE kernel.
        
        Args:
            body_name: Body name (e.g., 'MOON', 'SUN')
            time_jd: Time as Julian Date
            observer: Observer body (default 'EARTH')
            frame: Reference frame (default 'J2000')
            aberration: Aberration correction ('NONE', 'LT', 'LT+S')
        
        Returns:
            SPICEState with position (km) and velocity (km/s)
        
        Raises:
            RuntimeError: If kernels not loaded or query fails
        """
        if not self._kernels_loaded:
            raise RuntimeError(
                "SPICE kernels not loaded. Call load_kernels() first or "
                "check kernel directory and file availability."
            )
        
        try:
            # Convert JD to SPICE Ephemeris Time (ET)
            # SPICE expects UTC string or ET seconds. Using UTC ISO string.
            utc_string = self._jd_to_utc_string(time_jd)
            et = spice.str2et(utc_string)
            
            # Query state vector (position in km, velocity in km/s)
            body_id = self.BODY_IDS.get(body_name.upper(), body_name)
            observer_id = self.BODY_IDS.get(observer.upper(), observer)
            
            state_vector, lt = spice.spkezr(
                str(body_id),
                et,
                frame,
                aberration,
                str(observer_id)
            )
            
            # state_vector is [x, y, z, vx, vy, vz] in km and km/s
            position = np.array(state_vector[0:3])
            velocity = np.array(state_vector[3:6])
            
            return SPICEState(
                position=position,
                velocity=velocity,
                time_jd=time_jd,
                frame=frame,
                body_name=body_name
            )
        
        except Exception as e:
            raise RuntimeError(f"SPICE query failed for {body_name}: {e}")
    
    @staticmethod
    def _jd_to_utc_string(jd: float) -> str:
        """
        Convert Julian Date to UTC ISO string for SPICE.
        
        Args:
            jd: Julian Date
        
        Returns:
            UTC string (e.g., '2026-05-05T12:00:00')
        """
        # JD epoch: 2000-01-01 12:00 UT (JD 2451545.0)
        jd_epoch = 2451545.0
        mjd = jd - jd_epoch - 0.5  # Modified JD from epoch
        
        # Days since 2000-01-01
        days_since_epoch = jd - jd_epoch
        
        # Simple approximation: 2000-01-01 + days_since_epoch
        # For precision, use astropy if available
        try:
            from astropy.time import Time
            t = Time(jd, format='jd', scale='utc')
            return t.iso
        except ImportError:
            # Fallback: rough estimate
            # This is approximate; for production, use astropy
            import datetime
            epoch = datetime.datetime(2000, 1, 1, 12, 0, 0)
            delta = datetime.timedelta(days=days_since_epoch)
            dt = epoch + delta
            return dt.isoformat()
    
    def get_sun_position(self, time_jd: float) -> np.ndarray:
        """Get Sun position relative to Earth (km)."""
        state = self.get_body_state('SUN', time_jd, observer='EARTH')
        return state.position
    
    def get_moon_position(self, time_jd: float) -> np.ndarray:
        """Get Moon position relative to Earth (km)."""
        state = self.get_body_state('MOON', time_jd, observer='EARTH')
        return state.position
    
    def get_moon_velocity(self, time_jd: float) -> np.ndarray:
        """Get Moon velocity relative to Earth (km/s)."""
        state = self.get_body_state('MOON', time_jd, observer='EARTH')
        return state.velocity
    
    def __del__(self):
        """Cleanup: unload kernels if needed."""
        if SPICEYPY_AVAILABLE and self._kernels_loaded:
            try:
                spice.kclear()
            except Exception:
                pass  # Ignore cleanup errors
