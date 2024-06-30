# import the py-EM module and make its functions available
import pyEM as em
import sys
import numpy as np
import math as m
import json

#targetDefocus = -6.5 #in um

# load the navigator file
navfile = sys.argv[1]
navlines = em.loadtext(navfile)
allitems = em.fullnav(navlines)

# find the navigator items with active 'Acquire' flag
# first find items that have an 'Acquire' entry
acq = filter(lambda item:item.get('Acquire'),allitems)
acq = list(filter(lambda item:item['Acquire']==['1'],acq))

if not acq:
    raise RuntimeError('No "Acquire" flag has been set')

# Check that there is only one map with the "Acquire" flag
# otherwise stop execution

if len(acq) > 1:
    raise RuntimeError('There is more then 1 map with the Aquire flag')
stageZ = acq[0]['StageXYZ'][2]

# in this case, we have only one item
mapitem = acq[0]

#Extract informations from the merged map
mergedmap = em.mergemap(mapitem, blendmont=False)
idx_arr = np.array(mergedmap['sections'])
tilepixel_arr = np.array(mergedmap['tilepx1'])

# find the Registration of the map as the points shall go into the same
regis = mapitem['Regis']

# import the coordinates from json file
coordsLines = []
coordsFile = navfile[:-4] + '.json'
with open(coordsFile) as json_file:
    coordsData = json.load(json_file)
    for key in coordsData.keys():
        if coordsData[key]:
            for value_idx in np.arange(len(coordsData[key])):
                tilePiece = int(key[-4:])
                coordsLines.append(
                    str(coordsData[key][value_idx][0])
                    + ','
                    # Remember to change the 4096 to something not fix
                    + str(mergedmap['mapheader']['ysize'] - coordsData[key][value_idx][1])
                    + ','
                    + str(tilePiece)
                    )
    
#importlines =  em.loadtext(coordsFile)
importlines =  coordsLines

# create a navigator
newitems=list()

#remove the Acquire flag from the map item
mapitem['Acquire'] = '0'

newitems.append(mapitem)

addPointsPixel=[]
# fill the list with new point items
for idx,line in enumerate(importlines):
    #split the coordinates by the comma separator
    coords = line.split(',')
    pointPx=(tilepixel_arr[int(coords[2])][0]+float(coords[0]),tilepixel_arr[int(coords[2])][1]+float(coords[1]))

    if idx>0:        
        distance=[np.sqrt((pointPx[0]-p[0])**2 + (pointPx[1]-p[1])**2) for p in addPointsPixel]
        
        if np.min(distance)<400:
            print('Excluded point {} on section {} because it was too close ({} pixels to nearest point)'.format(idx+1,coords[2],np.min(distance)))
            continue
    
    #fill the relevant fields of the navigator item to create a point with the given label
    point = em.pointitem('Point_'+str(idx+1)+'_Section_'+coords[2],regis)    
    point['NumPts'] = str(0) #required for external coordinates 
    #add the information on which map to place the point
    point['DrawnID'] = mapitem['MapID']
    
    
    
    # generate grup IDs for points
    # points in the same tile will have same group ID
    if idx == 0:
        grpID = em.newID(allitems, 1)
        tileNum = int(coords[2])
    elif int(coords[2]) != tileNum:
        grpID = grpID + 1
        tileNum = int(coords[2])
        
    point['GrupID'] = str(grpID)
    #add the coordinates, we need an additional z-value for the CoordsInMap entry
    point['PieceOn'] = [str(idx_arr[int(coords[2])])]
    #add a zero to the coordinates for the Z coord
    coords[2] = stageZ
    point['CoordsInPiece'] = coords
    point['Acquire'] = str(0)
    
    newitems.append(point)
    addPointsPixel.append(pointPx)

# create new file by copying the header of the input file
newnavf = navfile[:-4] + '_with_points.nav'
nnf = open(newnavf,'w')
nnf.write("%s\n" % navlines[0])
nnf.write("%s\n" % navlines[1])

# fill the new file   
for nitem in newitems:
    out = em.itemtonav(nitem,nitem['# Item'])
    for item in out: nnf.write("%s\n" % item)
            
nnf.close()

# done