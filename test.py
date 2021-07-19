import math
from PIL import Image, ImageDraw
  
# w, h = 220, 190
# shape = [(40, 40), (w - 10, h - 10)]
  
# # creating new Image object
# img = Image.new("RGB", (w, h))
  
# # create rectangle image
# img1 = ImageDraw.Draw(img)  
# img1.rectangle(shape, fill ="#ffff33", outline ="red")
# img.show()

img = Image.open('./cats.jpg')
# img = img.resize((736, 384))

width = img.width
heigh = img.height

print('org: ', img.size)

start = (0.28763292, 0.09838788)
size = (0.25449764, 0.339828)
true_shape = [
        (start[0] * width, start[1] * heigh), 
        (start[0] * width + size[0] * width, start[1] * heigh + size[1] * heigh)
    ]


draw_img = ImageDraw.Draw(img)
draw_img.rectangle(true_shape, outline='yellow')

img.show()