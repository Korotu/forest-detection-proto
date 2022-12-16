#!/usr/bin/env python3

import cv2
import json
import geojson
import numpy as np
from geojson import Feature, Point, FeatureCollection, Polygon, dump

def getNESWextents(GeoJSONfile):

    # Load the enclosing rectangle JSON
    with open(GeoJSONfile,'r') as datafile:
        data = json.load(datafile)
    # print(data)
    # feature_collection = FeatureCollection(data['features'])

    lats = []
    lons = []
    for coords in data['geometry']['coordinates'][0]:
        # coords = data['geometry']['coordinates']
        lons.append(coords[0])
        lats.append(coords[1])

    # Work out N, E, S, W extents of boundaries
    Nextent = max(lats)
    Sextent = min(lats)
    Wextent = min(lons)
    Eextent = max(lons)
    return Nextent, Eextent, Sextent, Wextent

def getGreyscaleImg(imageFilePath):
    im = cv2.imread(imageFilePath)
    im_greyscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im_greyscale


def loadAndTrimImage(imagefilename):
    """Loads the named image and trims it to the extent of its content"""
    # Open shape image and extract alpha channel
    im = cv2.imread(imagefilename,cv2.IMREAD_UNCHANGED)
    print(np.shape(im))
    alpha = im[...,2]
    # Find where non-zero, i.e. not black
    y_nonzero, x_nonzero = np.nonzero(1)
    # Crop to extent of non-black pixels and return
    res = alpha[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    # Threshold to pure white on black
    # want to use the original image for edge detection
    _, res = cv2.threshold(res, 64, 255, cv2.THRESH_BINARY)
    return res

def getVerticesContours(im):
    """Gets the vertices of the shape in im"""
    # print(np.shape(cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)))
    contours = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    
    

    # area = cv2.contourArea(contours)
    # print(area)
    # Should probably sort by contour area here - and take contour with largest area
    perim = cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], 0.01 * perim, True)

    print(f"DEBUG: Found shape with {approx.shape[0]} vertices")
    return approx

def getVertices(im_greyscale):
    
    img_blur = cv2.GaussianBlur(im_greyscale, (9,9), sigmaX=1, sigmaY=1)

    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    # Crop to extent of non-black pixels and return
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.waitKey(0)
    
    (thresh, blackAndWhiteImage) = cv2.threshold(sobelxy, 127, 255, cv2.THRESH_BINARY)
    
    
    contours = cv2.findContours(blackAndWhiteImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    perim = cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], 0.01 * perim, True)
    print(f"DEBUG: Found shape with {approx.shape[0]} vertices from Sobel")

    
if __name__ == "__main__":

    # Get N, E, S, W extents from JSON file
    Nextent, Eextent, Sextent, Wextent = getNESWextents('./inventory/inventory/store/regions/boundary_FfxHJmH.geojson')
    print(f'DEBUG: Nextent={Nextent}, Eextent={Eextent}, Sextent={Sextent}, Wextent={Wextent}')

    # Load the image and crop to contents
    im = loadAndTrimImage('./image_polygon.png')
    im_greyscale = getGreyscaleImg('./image_polygon.png')
    print('DEBUG: Trimmed image is "trimmed.png"')
    cv2.imwrite('trimmed.png', im)
    cv2.imwrite('greyscale.png', im_greyscale)
    # Get width and height in pixels
    Hpx, Wpx = im.shape
    
    # Get width and height in degrees
    Hdeg, Wdeg = Nextent-Sextent, Eextent-Wextent
    # Calculate degrees per pixel in East-West and North-South direction
    degppEW = Wdeg/Wpx
    degppNS = Hdeg/Hpx
    print(f'DEBUG: degppEW={degppEW}, degppNS={degppNS}')

    # Get vertices of shape and stuff into list of features
    coordinates = []
    features = []
    vertices = getVerticesContours(im)
    vertices_sobel = getVertices(im_greyscale)
    first = (1, 1)
    for i in range(vertices.shape[0]):
       x, y = vertices[i,0]
       lon = Wextent + x*degppEW
       lat = Nextent - y*degppNS
       if i == 0:
           first = (lon, lat)
       print(f'DEBUG: Vertex {i}: imageX={x}, imageY={y}, lon={lon}, lat={lat}')
       coordinates.append((lon, lat))
    coordinates.append(first)
    coordinatesArr = []
    coordinatesArr.append(coordinates)
    polygon = Polygon(coordinatesArr)

    feature = Feature(geometry=polygon)
    features.append(feature)
    # Convert list of features into a FeatureCollection and write to disk
    featureCol = FeatureCollection(features)
    with open ('result.geojson', 'w') as f:
        dump(featureCol, f)