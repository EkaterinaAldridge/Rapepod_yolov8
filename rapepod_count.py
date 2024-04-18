import os
import json

if __name__ == '__main__':
    filepath = 'datasets/rapedata/json_labels'
    rape_count = 0
    file_count = 0
    for root, dirs, files in os.walk(filepath):
        for filename in files:
            with open(filepath + '/' +filename) as json_data:
                result = json.load(json_data)
                one_json_list = result.get('shapes')
                temp = len(one_json_list)
                print('The rapepod number of ' + filename + ' is', temp)
                rape_count = rape_count + temp
                file_count = file_count + 1
    print('The summary file number is', file_count)
    print('The summary rapepod number is', rape_count)