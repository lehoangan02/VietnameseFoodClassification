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

FREQUENCY_LIST_TRAIN = [0] * 30 # number of files in each folder
class FoodProcessor:
    filepath = ""
    OutputPath = "./"
    leaf = ""
    def __init__(self, filepath):
        self.filepath = filepath
        Leaf = os.path.basename(filepath)
        self.leaf = Leaf
        print(f"Leaf: {Leaf}")
    def processToCSV(self, FileName):
        with open (os.path.join(self.OutputPath, FileName), mode='w', newline='') as File:
            Writer = csv.writer(File)
            # Writer.writerow(['filename', 'label'])
            for root, dirs, files in os.walk(self.filepath):
                Label = os.path.basename(root)
                if Label in LABELS_LIST:
                    for Name in files:
                        Writer.writerow([Name, LABELS_INDEX[Label]])
                        # Writer.writerow([Name, Label])
    def calculateIndexOffset(self):
        global FREQUENCY_LIST_TRAIN, LABELS_LIST
        Directory = os.path.join(self.filepath)
        # print(f"File path: {Directory}")
        count = 0
        for index, label in enumerate(LABELS_LIST):
            Label_Directory = os.path.join(Directory, label)
            if os.path.exists(Label_Directory):
                for root, dirs, files in os.walk(Label_Directory):
                    for file in files:
                            count += 1
                FREQUENCY_LIST_TRAIN[index] = count
                # print(f"Number of files in the folder {LABELS_LIST[index]} is: {count}")
    def renameCumulativelyAndMove(self):
        global FREQUENCY_LIST_TRAIN
        Directory = os.path.join(self.leaf)
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
                        new_file_path = os.path.join(root, new_file_name)
                        shutil.copy(old_file_path, new_file_path)
    def processImageToDirectory(self):
        target_dir = os.path.join(self.OutputPath, self.leaf)
        print(f"Target directory: {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        for root, dirs, files in os.walk(self.filepath):
            for file in files:
                if file.startswith("._"):
                    continue
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                shutil.copy(src_path, dst_path)
    def removeErrorDataCSV(self, ErrorString, FileName):
        with open (os.path.join(self.OutputPath, FileName), mode='r') as File:
            Reader = csv.reader(File)
            FileName = FileName.split(".")[0]
            NewFileName = FileName + "Fixed.csv"
            with open (os.path.join(self.OutputPath, NewFileName), mode='w', newline='') as File:
                Writer = csv.writer(File)
                for row in Reader:
                    if not row[0].startswith(ErrorString):
                        Writer.writerow(row)
        print("Removed all errors.")
    def removeErrorDataDirectory(self, ErrorString):
        target_dir = os.path.join(self.OutputPath, self.leaf)
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
Processor = FoodProcessor("./archive/Images/Validate")
Processor.removeAllError("._")
Processor.calculateIndexOffset()
# print(FREQUENCY_LIST_TRAIN)
Processor.renameCumulativelyAndMove()
Processor.processToCSV("ValidateLabels.csv")

# Processor.removeErrorDataDirectory("._")