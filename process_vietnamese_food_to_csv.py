import os
import csv
import shutil

filepath = "./archive/Images"

LABELS_LIST = ["Banh beo", "Banh bot loc", "Banh can", "Banh canh",
            "Banh chung", "Banh cuon", "Banh duc", "Banh gio",
            "Banh khot", "Banh mi", "Banh pia", "Banh tet",
            "Banh trang nuong", "Banh xeo", "Bun bo Hue",
            "Bun dau mam tom", "Bun mam", "Bun rieu", "Bun thit nuong",
            "Ca kho to", "Canh chua", "Cao lau", "Chao long",
            "Com tam", "Goi cuon", "Hu tieu", "Mi quang", "Nem chua",
            "Pho", "Xoi xeo"]
LABELS_INDEX = {label: index for index, label in enumerate(LABELS_LIST)}
print(len(LABELS_INDEX))

FREQUENCY_LIST = [0] * 30 # number of files in each folder
class FoodProcessor:
    filepath = ""
    OUT_PATH = "./"
    leaf = ""
    def __init__(self, filepath):
        self.filepath = filepath
        Leaf = os.path.basename(filepath)
        self.leaf = Leaf
        print(f"Leaf: {Leaf}")
    def processToCSV(self, FileName):
        if not os.path.exists(os.path.join(self.OUT_PATH, self.leaf)):
            os.makedirs(self.OUT_PATH, self.leaf)
        if not os.path.exists(os.path.join(self.OUT_PATH, self.leaf, 'label')):
            os.makedirs(os.path.join(self.OUT_PATH, self.leaf, 'label'))
        with open (os.path.join(self.OUT_PATH, self.leaf, 'label', FileName), mode='w', newline='') as File:
            Writer = csv.writer(File)
            # Writer.writerow(['filename', 'label'])
            image_path = os.path.join(self.OUT_PATH, self.leaf, 'images')
            print(f"Image path: {image_path}")
            for root, dirs, files in os.walk(os.path.join(self.OUT_PATH, self.filepath)):
                # print(f"Root: {root}")
                Label = os.path.basename(root)
                # print(f"Label: {Label}")
                if Label in LABELS_LIST:
                    for Name in files:
                        Writer.writerow([Name, LABELS_INDEX[Label]])
    def calculateIndexOffset(self):
        global FREQUENCY_LIST, LABELS_LIST
        Directory = os.path.join(self.filepath)
        # print(f"File path: {Directory}")
        count = 0
        for index, label in enumerate(LABELS_LIST):
            Label_Directory = os.path.join(Directory, label)
            if os.path.exists(Label_Directory):
                for root, dirs, files in os.walk(Label_Directory):
                    for file in files:
                            count += 1
                FREQUENCY_LIST[index] = count
                # print(f"Number of files in the folder {LABELS_LIST[index]} is: {count}")
    def renameCumulativelyAndMove(self):
        global FREQUENCY_LIST
        Directory = os.path.join(self.filepath)
        NewFolder = os.path.join(self.OUT_PATH, self.leaf)
        if not os.path.exists(NewFolder):
            os.makedirs(NewFolder)
        if not os.path.exists(os.path.join(NewFolder, 'images')):
            os.makedirs(os.path.join(NewFolder, 'images'))
        print(f"File path: {Directory}")
        FileIndex = 1
        for index, label in enumerate(LABELS_LIST):
            Label_Directory = os.path.join(Directory, label)
            if os.path.exists(Label_Directory):
                for root, dirs, files in os.walk(Label_Directory):
                    for file in files:
                        old_file_path = os.path.join(root, file)
                        new_file_name = f"{FileIndex}.jpg"
                        FileIndex += 1
                        # print(f"Leaf: {self.leaf}")
                        new_file_path = os.path.join(self.OUT_PATH, self.leaf, 'images', new_file_name)
                        # print(f"Old file path: {old_file_path}")
                        # print(f"New file path: {new_file_path}")
                        shutil.copy(old_file_path, new_file_path)
    def __findLabel(self, FileName):
        file_index = FileName.split(".")[0]
        largest = 0
        for index, label in reversed(list(enumerate(FREQUENCY_LIST))):
            if FREQUENCY_LIST[index] > int(file_index):
                largest = index
        return largest
                
    def processCumulativeToCSV(self):
        if not os.path.exists(os.path.join(self.OUT_PATH, self.leaf)):
            os.makedirs(self.OUT_PATH, self.leaf)
        if not os.path.exists(os.path.join(self.OUT_PATH, self.leaf, 'label')):
            os.makedirs(os.path.join(self.OUT_PATH, self.leaf, 'label'))
        image_path = os.path.join(self.OUT_PATH, self.leaf, 'images')
        with open (os.path.join(self.OUT_PATH, self.leaf, 'label', 'labels.csv'), mode='w', newline='') as File:
            for root, dirs, files in os.walk(image_path):
                for file in files:
                    print(f"Processing {file}")
                    Writer = csv.writer(File)
                    label = self.__findLabel(file)
                    Writer.writerow([file, label])

    def processImageToDirectory(self):
        NewFolder = os.path.join(self.OUT_PATH, self.leaf)
        if not os.path.exists(NewFolder):
            os.makedirs(NewFolder)
        if not os.path.exists(os.path.join(NewFolder, 'images')):
            os.makedirs(os.path.join(NewFolder, 'images'))
        print(f"Target directory: {NewFolder}")
        os.makedirs(NewFolder, exist_ok=True)
        for root, dirs, files in os.walk(self.filepath):
            for file in files:
                if file == "11403.jpg":
                    print(f"Processing {file}")
                src_path = os.path.join(root, file)
                dst_path = os.path.join(NewFolder, 'images', file)
                shutil.copy(src_path, dst_path)
    def removeErrorDataCSV(self, ErrorString, FileName):
        with open (os.path.join(self.OUT_PATH, FileName), mode='r') as File:
            Reader = csv.reader(File)
            FileName = FileName.split(".")[0]
            NewFileName = FileName + "Fixed.csv"
            with open (os.path.join(self.OUT_PATH, NewFileName), mode='w', newline='') as File:
                Writer = csv.writer(File)
                for row in Reader:
                    if not row[0].startswith(ErrorString):
                        Writer.writerow(row)
        print("Removed all errors.")
    def removeErrorDataDirectory(self, ErrorString):
        target_dir = os.path.join(self.OUT_PATH, self.leaf)
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.startswith(ErrorString):
                    os.remove(os.path.join(root, file))
    def removeAllError(self, ErrorString):
        print(f"REMOVING FILES IN {self.filepath}")
        for root, dirs, files in os.walk(self.filepath):
            for file in files:
                if file.startswith(ErrorString):
                    os.remove(os.path.join(root, file))
Processor = FoodProcessor("./archive/Images/Test")
# Processor.removeAllError("._")
Processor.calculateIndexOffset()
print(FREQUENCY_LIST)
print(LABELS_INDEX)
# Processor.renameCumulativelyAndMove()
Processor.processCumulativeToCSV()
# Processor.processImageToDirectory()
# Processor.processToCSV("labels.csv")