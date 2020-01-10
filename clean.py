import tensorflow as tf
import main
import csv
import html
import re

def remove_html_tags_and_speachmarks(text):
    clean = re.compile('<.*?>|"')
    return re.sub(clean, '', text)

def remove_no_description(text):
    clean = re.compile('.*,,.*\n')
    return re.sub(clean, '', text)

def remove_no_title(text):
    clean = re.compile(',\n')
    return re.sub(clean, '', text)




with open("./argosProduct2020-01-09.csv", "r") as data_file:
    with open("./argos_reversed_title.csv","w" ) as clean_file:
        data = data_file.read()
        data = html.unescape(data)
        data = remove_no_description(data)
        data = remove_no_title(data)

        clean_file.write(remove_html_tags_and_speachmarks(data))
# run  awk -F, '{print$1,$3,$2}' OFS=, argos_reversed_title.csv  > argos_swaped.csv after



