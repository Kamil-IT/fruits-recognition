import os
from sklearn.model_selection import train_test_split
from PIL import Image

cwd = os.getcwd()

BASE_PATH = os.path.join(cwd, "dataset", "fruits_360_dataset", "fruits")

TRAIN_PATH = os.path.join(BASE_PATH, "Training")
TEST_PATH = os.path.join(BASE_PATH, "Test")


# def delete_background(img):
#     img = img.convert("RGBA")

#     datas = img.getdata()

#     newData = []

#     for item in datas:
#         if item[0] >= 225 and item[1] >= 225 and item[2] >= 225:
#             newData.append((255, 255, 255, 0))
#         else:
#             newData.append(item)
#     img.putdata(newData)

#     return img


# def save_in_new_dir(img, dirname, filename):
#     new_dir = dirname.replace("fruits_360_dataset", "without_background")
#     if not os.path.exists(new_dir):
#         os.makedirs(new_dir)
#     img.save(os.path.join(new_dir, filename), "PNG")


# for path_ in [TEST_PATH, TRAIN_PATH]:
#     for dirname, _, filenames in os.walk(path_):
#         for filename in filenames:
#             image = os.path.join(dirname, filename)
#             img = Image.open(image)
#             img = delete_background(img)
#             save_in_new_dir(img, dirname, filename)


for dirname, _, filenames in os.walk(os.path.join(cwd, "dataset", "without_background", "fruits", "Training")):
    if dirname.split('\\')[-1] != "Training":
        X_train, X_val= train_test_split(filenames, test_size=0.25)
        for element in X_val:
            image = os.path.join(dirname, element)
            img = Image.open(image)
            new_dir = dirname.replace("Training", "Validate")
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            img.save(os.path.join(new_dir, element), "PNG")
            os.remove(image) #dziala


            


        




        # save_in_new_dir(img, dirname, filename)
print ("end")

# def split_train():
