import xml.etree.ElementTree as ET
xml_path = "dataset/raw/extracted/annotations/maksssksksss0.xml"
tree=ET.parse(xml_path)
root = tree.getroot()
size = root.find('size')
width = int(size.find('width').text)
height = int(size.find('height').text)
print(width,height)
objects = root.findall('object')
classes = ["with_mask", "without_mask", "mask_weared_incorrect"]
for obj in objects:
    name = obj.find('name').text
    class_id = classes.index(name)
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    width_norm  = (xmax - xmin) / width
    height_norm = (ymax - ymin) / height
    center_x_norm = ((xmin + xmax) / 2) / width
    center_y_norm = ((ymin + ymax) / 2) / height
    print(class_id, center_x_norm, center_y_norm, width_norm, height_norm)