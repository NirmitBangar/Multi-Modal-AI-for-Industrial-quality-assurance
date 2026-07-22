"""
Week 1: Python Foundations for ML/AI Pipelines
================================================
This file covers core Python concepts that every ML engineer uses daily.
The examples are framed around industrial data — sensor readings, product batches,
defect logs — to stay connected to the project's end goal.

Key ideas practiced here:
  - Variables, data types, type coercion
  - Lists, tuples, dicts, sets (and when to use each)
  - Control flow: loops, comprehensions, conditionals
  - Functions: pure functions, default args, *args, **kwargs
  - Object-Oriented Programming: classes, encapsulation
  - File I/O and exception handling
"""

# =============================================================================
# SECTION 1: Data Types & Variables
# =============================================================================

# In Python, variables are just labels pointing to objects in memory.
# Type is inferred at runtime — no type declaration needed.

sensor_temperature = 73.5          # float — sensor reading in Celsius
product_id = 10234                 # int   — identifier
defect_detected = False            # bool  — binary QA outcome
product_label = "Metal Nut"        # str   — product category
no_reading = None                  # NoneType — absence of a value

# Type checking at runtime
print(type(sensor_temperature))    # <class 'float'>
print(type(defect_detected))       # <class 'bool'>

# Type coercion
# Python does NOT coerce silently between int/str (unlike JavaScript)
# This would raise a TypeError:
#   result = "Temp: " + sensor_temperature  ← ERROR
result = "Temp: " + str(sensor_temperature)   # Explicit cast required
print(result)  # Temp: 73.5


# =============================================================================
# SECTION 2: Collections — Lists, Tuples, Dicts, Sets
# =============================================================================

# --- List: Ordered, mutable sequence ---
# Used to store batches of sensor readings, image paths, etc.
sensor_readings = [72.1, 73.5, 74.0, 69.8, 80.3]

# Slicing: [start:stop:step]
recent_readings = sensor_readings[-3:]      # Last 3 readings: [74.0, 69.8, 80.3]
every_other = sensor_readings[::2]          # [72.1, 74.0, 80.3]

# List methods (in-place mutations)
sensor_readings.append(71.2)               # Add to end
sensor_readings.insert(0, 65.0)            # Insert at index 0
sensor_readings.sort()                     # Sort ascending (in-place)

print("Sorted readings:", sensor_readings)


# --- Tuple: Ordered, IMMUTABLE sequence ---
# Use when data should not change — e.g., image dimensions
image_shape = (256, 256, 3)   # (height, width, channels) — won't be modified
# image_shape[0] = 512        # TypeError: 'tuple' object does not support item assignment

# Tuple unpacking
height, width, channels = image_shape
print(f"Image: {height}×{width}, {channels} channels")


# --- Dictionary: Key-value mapping ---
# The workhorse of ML pipelines — config dicts, metadata stores, label maps

defect_catalog = {
    "scratch": {"severity": "high", "category": "surface", "code": "SC-01"},
    "dent":    {"severity": "medium", "category": "structural", "code": "DN-02"},
    "stain":   {"severity": "low", "category": "surface", "code": "ST-03"},
}

# Accessing nested dict
print(defect_catalog["scratch"]["severity"])   # "high"

# Safe access with .get() — won't raise KeyError if key is missing
unknown = defect_catalog.get("crack", "Not in catalog")
print(unknown)   # "Not in catalog"

# Iterating
for defect_type, details in defect_catalog.items():
    print(f"{defect_type}: severity={details['severity']}, code={details['code']}")


# --- Set: Unordered, unique elements ---
# Great for deduplication — e.g., unique defect codes seen in a batch
seen_codes = {"SC-01", "DN-02", "SC-01", "ST-03"}
print(seen_codes)  # {'SC-01', 'DN-02', 'ST-03'} — duplicates removed automatically

batch_a = {"SC-01", "DN-02"}
batch_b = {"DN-02", "ST-03", "CK-04"}
print("Defects in both batches:", batch_a & batch_b)    # Intersection: {'DN-02'}
print("All unique defects:", batch_a | batch_b)          # Union


# =============================================================================
# SECTION 3: Control Flow & Comprehensions
# =============================================================================

# --- Standard for loop ---
threshold = 75.0
defect_events = []
for reading in sensor_readings:
    if reading > threshold:
        defect_events.append(reading)

print("Readings above threshold:", defect_events)


# --- List comprehension: More Pythonic, slightly faster ---
# Same logic as above, in one line
defect_events_v2 = [r for r in sensor_readings if r > threshold]
print("Same result:", defect_events_v2)

# Comprehension with transformation
normalized = [round((r - min(sensor_readings)) / (max(sensor_readings) - min(sensor_readings)), 4)
              for r in sensor_readings]
print("Normalized readings:", normalized)


# --- Dict comprehension ---
# Map defect codes to their severity (inverting nested lookup)
severity_lookup = {
    details["code"]: details["severity"]
    for defect_type, details in defect_catalog.items()
}
print("Severity lookup:", severity_lookup)
# {'SC-01': 'high', 'DN-02': 'medium', 'ST-03': 'low'}


# =============================================================================
# SECTION 4: Functions — Pure, Reusable, Documented
# =============================================================================

def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Min-max normalization: scales value to the [0, 1] range.

    This is one of the most common preprocessing steps in ML.
    Without normalization, features with larger magnitudes dominate
    the loss function — e.g., a temperature feature (0–300°C) would
    overwhelm a binary defect label (0 or 1) in gradient descent.

    Args:
        value: The raw sensor reading.
        min_val: Minimum value seen in the dataset.
        max_val: Maximum value seen in the dataset.

    Returns:
        Normalized value in [0, 1].

    Raises:
        ValueError: If min_val == max_val (zero range → division by zero).
    """
    if min_val == max_val:
        raise ValueError("min_val and max_val are equal — normalization undefined.")
    return (value - min_val) / (max_val - min_val)


def classify_defect_severity(temperature: float,
                              vibration: float,
                              pressure: float,
                              temp_threshold: float = 75.0,
                              vibration_threshold: float = 2.5) -> str:
    """
    A simple rule-based defect classifier (what we had before ML!).

    This illustrates why ML is needed: manually setting thresholds is
    brittle — it doesn't generalize, requires expert tuning, and can't
    capture non-linear interactions between variables.

    Args:
        temperature: Sensor temperature in Celsius.
        vibration: Vibration amplitude in mm/s.
        pressure: Line pressure in bar.
        temp_threshold: Default alert temperature.
        vibration_threshold: Default alert vibration level.

    Returns:
        Severity string: "CRITICAL", "WARNING", or "NORMAL"
    """
    critical = (temperature > 90) or (vibration > 4.0) or (pressure > 15)
    warning = (temperature > temp_threshold) or (vibration > vibration_threshold)

    if critical:
        return "CRITICAL"
    elif warning:
        return "WARNING"
    else:
        return "NORMAL"


# --- *args and **kwargs: Flexible function signatures ---
def log_sensor_event(*readings: float, station: str = "Station-1", **metadata) -> dict:
    """
    Demonstrates *args (variable positional) and **kwargs (variable keyword).

    In ML pipelines, flexible interfaces let you log any number of
    sensor channels without changing the function signature.

    Returns:
        A structured log entry dict.
    """
    return {
        "station": station,
        "readings": list(readings),
        "avg": sum(readings) / len(readings),
        "metadata": metadata
    }

event = log_sensor_event(71.2, 73.5, 80.1, station="Station-A",
                          shift="morning", operator="Ravi")
print("\nSensor event log:")
for k, v in event.items():
    print(f"  {k}: {v}")


# Test the classifier
print("\nClassifier output:")
print(classify_defect_severity(temperature=92, vibration=1.2, pressure=8))   # CRITICAL
print(classify_defect_severity(temperature=77, vibration=1.2, pressure=8))   # WARNING
print(classify_defect_severity(temperature=68, vibration=1.0, pressure=6))   # NORMAL


# =============================================================================
# SECTION 5: Object-Oriented Programming — A QA Pipeline Class
# =============================================================================

class SensorBatch:
    """
    Represents a batch of sensor readings from the production line.

    Encapsulating data + behaviour in a class mirrors how real ML systems
    work — you don't pass raw arrays everywhere; you build objects that
    carry their own validation, normalization, and reporting logic.

    Attributes:
        batch_id (str): Unique identifier for this batch.
        readings (list[float]): Raw sensor values collected during the batch.
        _normalized (list[float] | None): Cached normalized readings.
    """

    def __init__(self, batch_id: str, readings: list[float]):
        self.batch_id = batch_id
        self.readings = readings
        self._normalized = None   # Lazy computation — computed only when needed

    def __len__(self) -> int:
        return len(self.readings)

    def __repr__(self) -> str:
        return f"SensorBatch(id={self.batch_id!r}, n={len(self)}, avg={self.mean():.2f})"

    def mean(self) -> float:
        return sum(self.readings) / len(self.readings)

    def std(self) -> float:
        """Standard deviation — key statistic for anomaly detection."""
        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self.readings) / len(self.readings)
        return variance ** 0.5

    def normalize(self) -> list[float]:
        """Min-max normalize the readings, cached after first computation."""
        if self._normalized is None:
            min_r, max_r = min(self.readings), max(self.readings)
            self._normalized = [normalize(r, min_r, max_r) for r in self.readings]
        return self._normalized

    def detect_anomalies(self, n_std: float = 2.0) -> list[tuple[int, float]]:
        """
        Statistical anomaly detection using z-score thresholding.

        Any reading more than `n_std` standard deviations from the mean
        is flagged. This is the simplest anomaly detector, and it's the
        conceptual ancestor of Isolation Forest, Autoencoders, etc.

        Args:
            n_std: Number of standard deviations to use as threshold.

        Returns:
            List of (index, value) tuples for anomalous readings.
        """
        mean = self.mean()
        std = self.std()
        return [(i, r) for i, r in enumerate(self.readings)
                if abs(r - mean) > n_std * std]

    def summary(self) -> dict:
        anomalies = self.detect_anomalies()
        return {
            "batch_id": self.batch_id,
            "count": len(self),
            "mean": round(self.mean(), 3),
            "std": round(self.std(), 3),
            "min": min(self.readings),
            "max": max(self.readings),
            "anomalies_detected": len(anomalies),
            "anomaly_indices": [i for i, _ in anomalies]
        }


# Demo
batch = SensorBatch("BATCH-20250601", [72.1, 73.5, 74.0, 69.8, 80.3, 95.1, 71.2, 68.5])
print("\n" + "="*50)
print("SensorBatch Demo")
print("="*50)
print(repr(batch))
print("\nSummary:", batch.summary())
print("Normalized:", [round(x, 4) for x in batch.normalize()])


# =============================================================================
# SECTION 6: Exception Handling — Robust ML Pipelines
# =============================================================================

def safe_read_sensor(raw_input) -> float:
    """
    Safely parse a sensor value from potentially dirty data.

    In real production systems, sensor data is messy: missing values,
    string artifacts from firmware bugs, overflow values, etc.
    Exception handling here prevents a single bad reading from crashing
    the entire inspection pipeline.
    """
    try:
        value = float(raw_input)
        if value < 0 or value > 200:
            raise ValueError(f"Sensor reading {value} out of valid range [0, 200]")
        return value
    except TypeError:
        print(f"[WARN] Cannot convert {raw_input!r} to float — skipping.")
        return None
    except ValueError as e:
        print(f"[WARN] Invalid value: {e} — skipping.")
        return None


# Test with messy data (common in real industrial systems)
raw_data = [72.5, "ERROR", None, 85.3, "200.0", 999, -1, 73.1]
cleaned = [v for raw in raw_data if (v := safe_read_sensor(raw)) is not None]
print("\nCleaned sensor data:", cleaned)
