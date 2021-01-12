import math

def azimuth(ln):
    """
    Get the azimuth of a Shapely LineString object.
    :param ln: Shapely LineString object
    :return: Azimuth in degrees
    """
    return math.degrees(math.atan2(
        (max(ln.xy[0][1], ln.xy[0][0]) - min(ln.xy[0][1], ln.xy[0][0])),
        (max(ln.xy[1][1], ln.xy[1][0]) - min(ln.xy[1][1], ln.xy[1][0]))
    ))
