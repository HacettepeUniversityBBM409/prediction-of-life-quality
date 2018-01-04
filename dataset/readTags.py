import pygeoj
import os
import numpy as np
import Mapzen
import sys
'''
To use this program you must install pygeoj library.And data files must be encoded with ANSI.You can use notepad++ or powershell command to change.
You must run ReadTags.py and 
'''
citypops=Mapzen.returnPopdict("data.csv")
Traincities={'aachen':90.52 ,'aberdeen':76.77 ,'abu-dhabi':80.8 ,'addis-abeba':28.41 ,'adelaide':91.54 ,'ahmedabad':57.01 ,'alicante':72.53 ,'amman':39.57 ,'amsterdam':72.85 ,'asheville':81.34 ,'atlanta':80.51 ,'auckland':66.98 ,'austin':86.51 ,'baku':17.2 ,'baltimore':82 ,'bangalore':52.11 ,'bangkok':37.54 ,'basel':88.27 ,'beijing':25.69 ,'belfast':80.55 ,'belgrad':35.69 ,'bergen':83.76 ,'boston':82.81 ,'bratislava':56.18 ,'brighton':62.62 ,'brisbane':77.07 ,'bristol':67.91 ,'brussels':65.42 ,'bucharest':29.45 ,'buenos-aires':39.78 ,'cairo':17.25 ,'cebu':41.21 ,'chiang-mai':37.5 ,'christchurch':76.6 ,'cologne':82.82 ,'colombo':26.68 ,'copenhagen':75.23 ,'cordoba':47.68 ,'cork':76.37 ,'curitiba':58.67 ,'dallas':80.56 ,'darwin':78.52 ,'davao':30.66 ,'delhi':42.45 ,'detroit':50.99 ,'dhaka':13.48 ,'doha':80.28 ,'dresden':90.21 ,'dubai':85.16 ,'dusseldorf':88.02 ,'edinburgh':75.03 ,'edmonton':89.58 ,'fortaleza':36.68 ,'frankfurt':88.04 ,'gaborone':31.77 ,'geneva':82.76 ,'glasgow':80 ,'gold-coast':81.31 ,'gothenburg':71.51 ,'haifa':52.08 ,'hamburg':87.18 ,'hamilton':84.83 ,'hanoi':5.29 ,'helsinki':75.85 ,'hong-kong':59.5 ,'honolulu':79.58 ,'houston':74.08 ,'hyderabad':54.97 ,'indianapolis':77.13 ,'indore':48.53 ,'istanbul':45.67 ,'jakarta':15.66 ,'jerusalem':50.87 ,'johannesburg':51.26 ,'karachi':18.5 ,'kharkiv':30.76 ,'koci':45.13 ,'kolkata':25.64 ,'lagos':19.48 ,'lahore':28.52 ,'las-vegas':60.5 ,'lausanne':73.21 ,'leeds':78.06 ,'lima':22.67 ,'limassol':74.83 ,'ljubljana':61.87 ,'lodz':46.29 ,'london':47.89 ,'lyon':71.13 ,'malmo':60.32 ,'manama':63.85 ,'manchester':73 ,'manila':13.14 ,'marbella':75.24 ,'medellin':39.7 ,'melbourne':68.99 ,'milan':40.8 ,'minneapolis-saint-paul':83.79 ,'montevideo':44.63 ,'montreal':78.55 ,'munich':90.08 ,'muscat':58.99 ,'naples':42.92 ,'new-york':61.94 ,'nice':68.34 ,'noida':42.7 ,'nottingham':75.88 ,'orlando':82.33 ,'oxford':72.09 ,'perth':74.62 ,'philadelphia':65.53 ,'phoenix':82.55 ,'porto-alegre':31.87 ,'prague':58.85 ,'pune':43.84 ,'quito':46.43 ,'regina':79.98 ,'rio-de-janeiro':21.32 ,'riyadh':67.72 ,'rochester':78.05 ,'rome':26.8 ,'rotterdam':67.77 ,'saint-louis':87.51 ,'saint-petersburg':25.97 ,'san-antonio':84.88 ,'san-francisco':79.08 ,'santiago':39.13 ,'santo-domingo':41.88 ,'sao-paulo':30.57 ,'seattle':84.1 ,'seoul':60.28 ,'singapore':42.65 ,'sliema':65.03 ,'strasbourg':66.5 ,'sydney':74.32 ,'taipei':52.35 ,'tampa':80.75 ,'the-hague':79.99 ,'thessaloniki':43.61 ,'tirane':36.81 ,'tokyo':69.29 ,'vadodara':54.98 ,'vancouver':71.89 ,'genava':70.34 ,'vienna':77.21 ,'vilnus':64.19 ,'warsaw':51.82 ,'washington':76.92 ,'zagreb':47.59 ,'zurich':97.91}
Validation={'antalya':40.51 ,'athens':30.57 ,'barcelona':47.18 ,'belo-horizonte':36.26 ,'birmingham':75.4 ,'braga':69.91 ,'calgary':87.57 ,'canberra':93.05 ,'caracas':8.61 ,'casablanca':48.69 ,'charlotte':84.39 ,'chennai':43.89 ,'dublin':65.27 ,'galway':77.85 ,'gurgaon':42.77 ,'kyiv':24.05 ,'kuala-lumpur':42.4 ,'lisbon':56.08 ,'liverpool':83.14 ,'los-angeles':62.82 ,'luksemburg':81.41 ,'madrid':59.87 ,'malaga':67.9 ,'mexico-city':27.91 ,'miami':74.77 ,'minsk':51.66 ,'moscow':18.65 ,'mumbai':26.43 ,'nairobi':14.36 ,'nashville':80.5 ,'newark':73.21 ,'nicosia':57.58 ,'oslo':71.27 ,'ottawa':86.11 ,'paris':52.62 ,'pattaya':46.01 ,'portland':88.76 ,'porto':58.95 ,'pretoria':61.44 ,'shanghai':31.66 ,'sofia':37.72 ,'stavanger':76.5 ,'stockholm':78.58 ,'stuttgart':90.4 ,'toronto':77.02 ,'toulouse':66.66 ,'trondheim':82.67 ,'utrecht':72.61 ,'victoria':80.87 ,'wellington':79.83}
Test={'berlin':91.17 ,'bogota':29.33 ,'budapest':46.58 ,'cambridge':70.61 ,'cape-town':78.73 ,'chicago':80.71 ,'florence':53.73 ,'izmir':42.9 ,'leicester':76.19 ,'marseille':59.87 ,'new-orleans':81.2 ,'san-diego':80.18 ,'tehran':14.33 ,'ulaanbaatar':24.5 ,'valencia':64.89 ,'varna':36.03}
rootpath='D:\\ML PROJE\\Mapzen\\Test\\'#Change rootpath for creat from different data
path=rootpath
folders=os.listdir(rootpath)
CityArray=[]
cityScores=[]
i=0
CityNameArray=[]
print(folders)
for folder in folders:
    CityNameArray.append(folder.split('_')[0])
print(CityNameArray)
for name in CityNameArray:
    cityScores.append(float(Test[name]))#Change Dictionary for different data
print(cityScores)
np.save("testScore.npy",np.array(cityScores))#change filename for different data
for folder in folders[:]:
    path=rootpath+folder
    files=os.listdir(path)
    print(path.split('_')[0])
    ameneties,buildings,landusages,places,roads,transPoints,pools=[],[],[],[],[],[],[]
    i=0
    population=0
    scoreArray=[]
    for file in files[:]:
        file=path+'\\'+file
        try:
            test=pygeoj.load(filepath=file)
        except StopIteration:
            for i in range(0,500):
                print(StopIteration)
            pass
        except:
            print("--ERROR--:",file,"-->",sys.exc_info())
            continue
        print(file)
        '''the files is read by their names '''
        if(file.endswith("amenities.geojson")):#taking features
            ameneties=Mapzen.returnAmenitiesNum(test)
            print(len(ameneties),"Amenities:",ameneties)
        if (file.endswith("buildings.geojson")):
            buildings=Mapzen.returnBuildingsPercent(test)
            print(len(buildings),"buildings:",buildings)
        if (file.endswith("landusages.geojson")):
            landusages=Mapzen.returnLandUsages(test)
            print(len(landusages),"landusages:",landusages)
        if (file.endswith("places.geojson")):
            places=Mapzen.returnPlaces(test)
            print(len(places),"places:",places)
        if (file.endswith("roads.geojson")):
            roads=Mapzen.returnRoads(test)
            print(len(roads),"roads:",roads)
        if (file.endswith("transport_points.geojson")):
            transPoints=Mapzen.returnTransportPoints(test)
            print(len(transPoints),"transPoints:",transPoints)
        if (file.endswith("waterareas.geojson")):
            pools=Mapzen.returnPools(test)
            print(len(pools),"pools:",pools)
    print("population:",population)#creating one array
    scoreArray=Mapzen.divedePopulation(ameneties,citypops[folder.split('_')[0]])+buildings+Mapzen.divedePopulation(landusages[:23],citypops[folder.split('_')[0]])+landusages[23:]+places+Mapzen.divedePopulation(roads,citypops[folder.split('_')[0]])+Mapzen.divedePopulation(transPoints,citypops[folder.split('_')[0]])+Mapzen.divedePopulation(pools,citypops[folder.split('_')[0]])
    if(len(scoreArray)!=85):
        print(len(scoreArray),file)#printing to check
    CityArray.append(np.array(scoreArray))#append features of city to list
CityArray=np.array(CityArray)
np.save("test.npy",CityArray)