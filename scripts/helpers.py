import math

def calculate_angle(p1, p2):
    """Calculate the angle between two points."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    return angle
