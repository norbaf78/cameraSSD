#!/usr/bin/python
import math
import json

class CoordinateTransform:
    _log = None
    mPhi = 0.0
    mScale = 1.0
    mScalingErrorCorrection = 1.0
    aXReal = 0#1227753.06
    aYReal = 0#5762835.98
    aXVirtual = 0.0
    aYVirtual = 0.0
    bXReal = 0#1227751.256
    bYReal = 0#5762883.424
    bXVirtual = 0#30
    bYVirtual = 0#14
    mLatitude = 45.8908
	

    def __init__(self):
        with open('configReal.json') as json_data_file:
            data = json.load(json_data_file)  	
        #configuration transformation parameters
        self.aXReal= float(data['config']["pt_A_real_X"])
        self.aYReal= float(data['config']["pt_A_real_Y"])
        self.aXVirtual= float(data['config']["pt_A_virtual_X"])
        self.aYVirtual= float(data['config']["pt_A_virtual_Y"])
        self.bXReal= float(data['config']["pt_B_real_X"])
        self.bYReal= float(data['config']["pt_B_real_Y"])
        self.bXVirtual= float(data['config']["pt_B_virtual_X"])
        self.bYVirtual= float(data['config']["pt_B_virtual_Y"])
        self.mLatitude= float(data['config']["latitude"])		

        #Calculate temp values
        deltaRy = self.bYReal - self.aYReal
        deltaRx = self.bXReal - self.aXReal
        aR = math.atan2(deltaRy,deltaRx)
        deltaVy = self.bYVirtual - self.aYVirtual
        deltaVx = self.bXVirtual - self.aXVirtual
        aV = math.atan2(deltaVy,deltaVx)
        #set values
        self.mPhi = aR - aV
        self.mScale = self.getScalingFactor(self.mLatitude)
        #calculate distance discrepancy
        distanceReal = math.sqrt(pow((self.bXReal - self.aXReal), 2) + pow((self.bYReal - self.aYReal), 2))
        distanceVirtual = math.sqrt(pow((self.bXVirtual - self.aXVirtual), 2) + pow((self.bYVirtual - self.aYVirtual), 2))
        self.mScalingErrorCorrection = self.getScaleCorrection(distanceReal, distanceVirtual, self.mScale)       

    def getScalingFactor(self,latitude):
        self.mScale = 1 / math.cos(math.radians(latitude))
        return self.mScale

    def getScaleCorrection(self,distanceReal, distanceVirtual, scalingFactor):
        return (distanceReal / (distanceVirtual * scalingFactor))

    def transform(self,x,y):
        global _log
        xscaled = (x - self.aXVirtual) * self.mScale * self.mScalingErrorCorrection
        yscaled = (y - self.aYVirtual) * self.mScale * self.mScalingErrorCorrection       
        #transform
        tx = (xscaled * math.cos(self.mPhi)) - (yscaled * math.sin(self.mPhi)) + self.aXReal;
        ty = (xscaled * math.sin(self.mPhi)) + (yscaled * math.cos(self.mPhi)) + self.aYReal;
        return tx,ty    
    
    def pixelToMeter(self,x,y,maxPixelX,maxPixelY):
        xmeters = (self.bXVirtual / maxPixelX) * x
        ymeters = (self.bYVirtual / maxPixelY) * y # (0,0) on the upper left corner
        #ymeters = self.bYVirtual - (self.bYVirtual / maxPixelY) * y # (0,0) on the bottom left corner
        return xmeters,ymeters      

    def meterToPixel(self,xmeters,ymeters,maxPixelX,maxPixelY):
        x = (maxPixelX / self.bXVirtual) * xmeters;
        y = (maxPixelY / self.bYVirtual) * ymeters;
        return x,y
    
    def inverseTransform(self,tx,ty):
        yscaled = (math.cos(self.mPhi) * (ty - self.aYReal)) - (math.sin(self.mPhi) * (tx - self.aXReal))
        xscaled = (math.sin(self.mPhi) * (ty - self.aYReal)) + (math.cos(self.mPhi) * (tx - self.aXReal))
        coordX = xscaled / (self.mScale * self.mScalingErrorCorrection) + self.aXVirtual;
        coordY = yscaled / (self.mScale * self.mScalingErrorCorrection) + self.aYVirtual;
        return coordX,coordY

