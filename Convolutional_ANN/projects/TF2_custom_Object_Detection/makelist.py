# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:22:25 2024

@author: ismt
"""

'''
Use this script to generate a list of all XML files in a folder.
'''

from glob import glob

files = glob('*.xml')
with open('xml_list.txt', 'w') as f:
  for fn in files:
    f.write("%s\n" % fn)