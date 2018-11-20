#!/usr/bin/env python
import math

class ReprojectTo4326:
    originShift = 0

    def __init__(self):
        global originShift
        originShift = 2 * math.pi * 6378137 / 2.0
        #print("originshift: " + str(originShift))
        # 20037508.342789244

    def MetersToLatLon(self, mx, my ):
        "Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum"

        lon = (mx / originShift) * 180.0
        lat = (my / originShift) * 180.0

        lat = 180 / math.pi * (2 * math.atan( math.exp( lat * math.pi / 180.0)) - math.pi / 2.0)
        return lat, lon

    def latLonToMeters(self, lat, lon):
        "Converts lat/lon in WGS84 Datum to Spherical Mercator EPSG:900913"
        
        lat = (math.log(math.tan(((lat * math.pi / 180) + math.pi / 2.0) / 2.0))) * 180.0 / math.pi
        
        mx = (lon / 180.0) * originShift
        my = (lat / 180.0) * originShift
        return mx, my

