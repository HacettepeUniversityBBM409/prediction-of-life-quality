import pygeoj
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

def returnPopdict(path):
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        citypops = dict()
        for row in reader:
            citypops[row.get('Name')] = int(row.get("2017 Population"))/1000
    return citypops

def returnCordinates(path):
    files = os.listdir(path)
    LongtitudeList = []
    LatitudeList = []
    for file in files[1:]:
        if file.endswith("housenumbers.geojson"):  # four geojson files skiped,this line will be change
            continue
        file = path + '\\' + file
        try:
            test = pygeoj.load(filepath=file)
        except:
            continue
        citydict = dict()
        # print("++++++++++++++",file,"+++++++++++++")#I'll print to see contents in file,it will be deleted
        for feature in test:
            coordinates = feature.geometry.coordinates
            if feature.geometry.type == "Point":
                LongtitudeList.append(coordinates[0])
                LatitudeList.append(coordinates[1])
            elif feature.geometry.type == "LineString":
                for coordinate in coordinates:
                    LongtitudeList.append(coordinate[0])
                    LatitudeList.append(coordinate[1])
            elif feature.geometry.type == "Polygon":
                for coordinate in coordinates[0]:
                    LongtitudeList.append(coordinate[0])
                    LatitudeList.append(coordinate[1])
    return [np.min(LongtitudeList),np.max(LongtitudeList),np.min(LatitudeList),np.max(LatitudeList)]
def isbetweenCordinares(coor0,coor1,feature):
    print(feature.geometry.coordinates)

def returnAmenitiesNum(structure):
    featureDict={'library':0,'fuel':0,'hospital':0,'police':0,'school':0,'townhall':0,'university':0,'fire_station':0}
    for feature in structure:
        if featureDict.get(feature.properties['type']) is not None:
            featureDict[feature.properties['type']]+=1
    return list(featureDict.values())

def returnBuildingsPercent(structure):#it return percent of buildings
    featureDict = {'church':0, 'industrial':0, 'apartments':0, 'university':0, 'office':0, 'house':0, 'residential':0, 'school':0,
                     'palace':0, 'terrace':0, 'public':0, 'commercial':0, 'train_station':0, 'shed':0, 'hospital':0, 'temple':0,
                     'retail':0, 'chapel':0, 'college':0, 'detached':0, 'garage':0,'mosque':0}
    for feature in structure:
        if featureDict.get(feature.properties['type']) is not None:
            featureDict[feature.properties['type']]+=1
    featureDict['religious_structure']=featureDict['church']+featureDict['temple']+featureDict['chapel']+featureDict['mosque']#religious structures are one paremater
    del featureDict['church']
    del featureDict['temple']
    del featureDict['chapel']
    del featureDict['mosque']
    return list(( list( featureDict.values() ) /np.sum( list( featureDict.values() ) )    )*100)

def returnLandUsages(structure):#returns number of places,area ratio of some values
    featureDict={'zoo':0, 'pedestrian':0, 'sports_centre':0, 'university':0,'park':0, 'stadium':0, 'school':0, 'place_of_worship':0, 'fuel':0, 'parking':0, 'grass':0, 'pitch':0, 'footway':0, 'theatre':0,'meadow':0,'retail':0, 'library':0, 'playground':0, 'hospital':0, 'heath':0,'college':0,'cinema':0,'pier':0,'forest':0, 'residential':0, 'golf_course':0, 'commercial':0,'scrub':0,'railway':0, 'farmyard':0,'farmland':0,"water-areas":0}
    withoutnameList = ['pedestrian', 'sports_centre','park', 'stadium', 'school','place_of_worship', 'fuel', 'parking', 'grass', 'pitch', 'footway', 'theatre', 'meadow', 'forest','residential', 'retail', 'library','playground', 'heath','commercial', 'scrub', 'railway', 'college', 'farmyard', 'allotments', 'farmland', 'cinema','pier']
    areaList = [ 'forest','residential','golf_course','commercial','scrub', 'railway','farmyard','farmland','water-areas']
    totalArea = 0
    for feature in structure:
        if featureDict.get(feature.properties['type']) is not None:
            if feature.properties['type'] in withoutnameList:
                if feature.properties['type'] in  areaList:
                    featureDict[feature.properties['type']]+=float(feature.properties['area'])
                    totalArea+=float(feature.properties['area'])
                else:
                    featureDict[feature.properties['type']]+=1
            elif feature.properties['name'] is not None:
                if feature.properties['type'] in  areaList:
                    featureDict[feature.properties['type']] += float(feature.properties['area'])
                    totalArea += float(feature.properties['area'])
                else:
                    featureDict[feature.properties['type']]+=1
    for key in areaList:
        try:
            featureDict[key]=(featureDict[key]/totalArea)*100
        except:
            continue
    return list(featureDict.values())

def returnPlaces(structure):#it returns ratio of populotion in city,village,suburb
    featureDict={'city':0, 'village':0, 'suburb':0,'town':0}
    for feature in structure:
        if featureDict.get(feature.properties['type']) is not None and feature.properties['population'] is not None:
            featureDict[feature.properties['type']]+=int(feature.properties['population'])
    populationList=list((list(featureDict.values())/np.sum(list(featureDict.values())))*100)
    populationList.append(np.sum(list(featureDict.values()))/500000)#divide population 500000 for comporison other cities
    return  populationList

def returnRoads(structure):#it returns length of roads
    featureDict ={'living_street':0, 'pedestrian':0, 'steps':0, 'trunk_link':0, 'path':0, 'rail':0, 'subway':0, 'cycleway':0, 'pier':0,'light_rail':0, 'funicular':0, 'raceway':0, 'tram':0}
    for feature in structure:
        if featureDict.get(feature.properties['type']) is not None:
            featureDict[feature.properties['type']]+=len(feature.geometry.coordinates)
    return list(featureDict.values())

def returnTransportPoints(structure):#it returns number of points
    featureDict={'subway_entrance':0,'bus_stop':0, 'station':0,'crossing':0, 'helipad':0, 'tram_stop':0}
    for feature in structure:
        if featureDict.get(feature.properties['type']) is not None:
            if feature.properties['type'] == 'crossing':
                featureDict['crossing'] += 1
            elif feature.properties['name'] is not None:
                featureDict[feature.properties['type']]+=1
    return  list(featureDict.values())

def returnTrasportAreas(structure):#returns number of aerodrome
    featureDict={'aerodrome':0}
    for feature in structure:
        if feature.properties['type']=='aerodrome':
            if feature.properties['name'] is not None:
                featureDict['aerodrome']+=1
    return list(featureDict.values())

def returnPools(structure):
    featureDict={'swimming_pool':0,'private_pool':0}
    for feature in structure:
        if feature.properties['type']=='swimming_pool':
            if feature.properties['name'] is not None:#private pool is not named.
                featureDict['swimming_pool']+=1
            else:
                featureDict['private_pool']+=1
    return  list(featureDict.values())

def divedePopulation(values,population):#this methods returns features per population
    if (population==0) or population is None:
        population=50000
    for i in range (0,len(values)):
        values[i]=values[i]/(population/10000)
    return values

def returnPopulation(structure):#it returns population of cities
    featureDict = {'city': 0, 'village': 0, 'suburb': 0, 'town': 0}
    population=0
    for feature in structure:
        if feature.properties['population'] is not None and featureDict.get(feature.properties['type']) is not None:
            if(feature.properties['type']=='city' or feature.properties['type']=='village'):
                population+=feature.properties['population']
    return population