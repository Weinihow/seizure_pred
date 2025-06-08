from tkinter.filedialog import askopenfilename
import pathlib
import csv
import os

path = askopenfilename()
file_name = os.path.basename(path)
file_name = os.path.splitext(file_name)[0]
# fine current directory
dir = pathlib.Path(__file__).parent.resolve()

with open(path, 'r') as txtfile:
    lines = txtfile.readlines()

data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
data_lines = data_lines[1:]

output = str(dir) + "\\" + file_name

fieldnames = ['time', 'CH1_LEFT', 'CH1_RIGHT', 'CH2_LEFT', 'CH2_RIGHT']

with open(output + ".csv", 'w', newline='') as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    csv_writer.writeheader()
    writer = csv.writer(csvfile)
    for line in data_lines:
        # Split by whitespace or tabs — adjust as needed
        row = line.split()
        writer.writerow(row)

print(output)