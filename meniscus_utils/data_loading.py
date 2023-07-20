
from xml.etree import ElementTree
#import glob
import pathlib
import numpy as np
from skimage import draw, io

def get_label_imgs(annote="proboscis_and_meniscus_trainInP.xml"):
    tree = ElementTree.ElementTree(file=annote)
    #folder = pathlib.Path(img_folder)
    #image_paths = folder.glob("*.png")
    image_annotes = tree.findall("image")
    #locs = {}
    masks = {}
    for img_ann in image_annotes:
        img_attribs = img_ann.attrib
        name = img_attribs['name']
        width = int(img_attribs['width'])
        height = int(img_attribs['height'])
        empty = np.zeros((height, width), dtype=bool)
        # io.imshow(empty)
        # io.show()
        men = img_ann.find("points")
        if men is None:
            masks[name] = empty
            continue
        men_attribs = men.attrib
        if men_attribs['label'] != 'MeniscusCenter':
            print("what.")
        pointstr = men_attribs['points']
        xstr,ystr = pointstr.split(',')
        col = float(xstr)
        row = float(ystr)
        
        mincol = max(0, round(col - 20))
        maxcol = min(width - 1, round(col + 20))
        minrow = max(0, round(row - 5))
        maxrow = min(height - 1, round(row + 5))

        empty[minrow:maxrow, mincol:maxcol] = True

        # io.imshow(empty)
        # io.show()
        
        masks[name] = empty
    return masks
        
def get_input_images(folder='trainInP'):
    directory = pathlib.Path(folder)
    image_names = directory.glob("*.png")
    inputs = {}
    for name in image_names:
        array = io.imread(str(name))
        inputs[str(name)] = array
    return inputs

def load_dset(annotations="proboscis_and_meniscus_trainInP.xml", folder='trainInP'):
    label_d = get_label_imgs(annotations)
    inp_d = get_input_images(folder)
    img_names = set(label_d) & set(inp_d)
    inputs = []
    labels = []
    for name in img_names:
        in_img = inp_d[name]
        lbl_img = label_d[name]
        inputs.append(in_img)
        labels.append(lbl_img)
    return np.array(inputs), np.array(labels)

