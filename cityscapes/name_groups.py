#!/bin/python
import os
import glob
import csv

MODES = [ 'train', 'test', 'val' ]
# mode = MODES[0]

def get_color(s):
  return s.replace("leftImg8bit", "gtFine").replace(".png","_color.png")

def get_labelIds(s):
  return s.replace("leftImg8bit", "gtFine").replace(".png","_labelIds.png")

def get_image_groups(img_files):
  return zip(img_files,
            [ get_color(s) for s in  img_files ],
            [ get_labelIds(s) for s in img_files ])

mode_csv = '{}_name_groups.csv'
for mode in MODES:
  img_files = glob.glob(os.path.join("leftImg8bit", mode, "*/*png"))
  with open(mode_csv.format(mode),'wb') as f_csv:
    mywriter = csv.writer(f_csv)
    for row in get_image_groups(img_files):
      mywriter.writerow(row)


