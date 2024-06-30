# import the py-EM module and make its functions available
import pyEM as em
import sys
import os
import numpy as np

navfile = sys.argv[1]
coordFile = sys.argv[2]

navlines = em.loadtext(navfile)
allitems = em.fullnav(navlines)
#print(allitems[1])

# find the navigator items with active 'Acquire' flag
# first find items that have an 'Acquire' entry
acq = filter(lambda item:item.get('Acquire'),allitems)
acq = list(filter(lambda item:item['Acquire']==['1'],acq))

# in this case, we have only one item
mapitem = acq[0]

# Extract informations from the merged map
mergedmap = em.mergemap(mapitem, blendmont=False)
# print(mergedmap)

# Overlap in px of each tile
tiles_overlap_x = mergedmap['overlap'][0]
tiles_overlap_y = mergedmap['overlap'][1]
# Size of single tiles minus the overlap
tile_size_x = mergedmap['mapheader']['xsize'] - tiles_overlap_x
tile_size_y = mergedmap['mapheader']['ysize'] - tiles_overlap_y
pixelsize_um = mergedmap['mapheader']['pixelsize']
# Array with the coordinates of each tile in px
# whith the origin in the lower left corner
tilepx = np.array(mergedmap['tilepx1'])

# coordinates of the region of interest found in IMOD
coords = np.loadtxt(coordFile, delimiter=',', comments='#')
#print(coords.shape)

# list containing the number of the tiles (ZValue) of the 
# regions of interest
tilenumbers = []

# loop over all points of interest and find the ZValue
for i in np.arange(coords.shape[0]):
    # round to the lower integer and multiply by the image size
    # to get the bottom left coordinates of the tile
    tile_orig_x = int(coords[i, 0] / tile_size_x) * tile_size_x
    tile_orig_y = int(coords[i, 1] / tile_size_y) * tile_size_y
       
    diffx = tilepx[:, 0] - tile_orig_x
    diffy = tilepx[:, 1] - tile_orig_y
    summed_diffs = diffx + diffy
    # find where the difference of the coordinates is equal to 0
    # meaning that that the index of the tile!
    try:
        tilenumbers.append(np.argwhere(summed_diffs == 0)[0, 0])
        #print(tilenumbers)
    except IndexError:
        print(tile_orig_x, tile_orig_y)
        print('The origin of the tile with coord {} at line number {} was not found.'.format(coords[i, :], i))
        tilenumbers.append(-1)
        #print(np.argwhere(summed_diffs == 0))
            
    
tilenumbers = np.array(tilenumbers)
outarray = np.column_stack((coords, tilenumbers))
outarray.astype(np.int16)

out_file = os.path.abspath(coordFile)[:-4]
out_file = out_file + '_with_tiles_numbers.txt'
np.savetxt(out_file, outarray, delimiter=',', fmt='%i')