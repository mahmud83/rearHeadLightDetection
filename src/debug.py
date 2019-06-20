python
demo.py
ctdet - -demo
webcam - -load_model.. / models / ctdet_coco_dla_2x.pth

heads
{'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}

heads
{'hm': 80, 'wh': 2, 'reg': 2}


class Animal:
    def __init__(self, animal_list):
        self.animals_name = animal_list

    def __getitem__(self, index):
        print(index)
        return self.animals_name[index]


animals = Animal(["dog", "cat", "fish"])
for animal in animals:
    print(animal)
