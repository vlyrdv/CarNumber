import time

from CarNumbers.start import EditImage

result = EditImage()

for i in range(1, 13):
    print(i)
    result.draw_normal_img(f"{i}.jpg", max_power=True)
    result.edit_form()


