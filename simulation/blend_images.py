from PIL import Image
import os 



file = "inf_only_32"
# 75, 31, 32
f1 = "repeats/" + file + "/0.png"
Image.open(f1).save("out.png")

for i in range(1, 10):
    f2 = "repeats/" + file + "/" + str(i) + ".png"
    i1 = Image.open("out.png")
    i2 = Image.open(f2)

    Image.blend(i1, i2, .5).save("out.png")
