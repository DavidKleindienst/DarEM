# -*- coding: utf-8 -*-
# import the py-EM module and make its functions available
import pyEM as em

# parse command line parameters

import sys

navfile = sys.argv[1]


# load the navigator file
navlines = em.loadtext(navfile)
allitems = em.fullnav(navlines)

# create a new list of navigator entries using the ordernav function
newnav = em.ordernav(allitems)

# create new file by copying the header of the input file
newnavf = navfile[:-4] + '_sorted.nav'

print('Navigator file was sorted and output is written as: ' + newnavf)

em.write_navfile(newnavf,newnav,xml=False)
    
