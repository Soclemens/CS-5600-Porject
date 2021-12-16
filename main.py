from tfl_image_anns import *

img_ann = make_image_ann_model()
train_tfl_image_ann_model(img_ann, train_x, train_y, test_x, test_y)
img_ann.save("test_run.tfl")
print(validate_tfl_image_ann_model(img_ann, validate_x, validate_y))