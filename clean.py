import tensorflow as tf
import main
import csv
import html
import re

def regex_clean(text):
    # this cleans all 
    # html tags
    # rows with no description
    # rows with no title

    clean = re.compile('<.*?>|(.*,,.*\n)|(,\n)|"')
    return re.sub(clean, '', text)

def normalise_rows(data):
    print("before {}".format(len(data)))
    column_num = len(data[0].split(","))
    new_data = []
    for row in data:
        parts = row.split(",")
        if len(parts) > column_num:
            description = ", ".join(parts[1:len(parts) - 1])
            new_row = [parts[0], parts[len(parts) - 1],description ]
            new_data.append("\t".join(new_row))
        elif len(parts) == column_num:
            new_row = [parts[0], parts[len(parts) - 1], parts[1]]
            new_data.append( "\t".join(parts))
    print("after {}".format(len(new_data)))
    return new_data



with open("./argosProduct2020-01-09.csv", "r") as data_file:
    with open("./argos.csv","w" ) as clean_file:
        regex_cleaned =  regex_clean(data_file.read())
        text_cleaned = html.unescape(regex_cleaned)
        text_as_csv  = text_cleaned.split("\n")
        text_as_csv = normalise_rows(text_as_csv)
        clean_file.write("\n".join(text_as_csv))
# run 
#  awk -F, '{print$1,$3,$2}' OFS=, argos_reversed_title.csv  > argos.csv
# after



