import numpy as np
def save_np_array(array,labels,file_name):
    """
    This functions aims at saving our array so as to keep our information.
    Params.
    ----
        array : np.array
            numpy array containing data.
        labels : array of str
            Contains the name of the coloumn of our arrays.
        file_name : str
            Name of the file to create and save. 
    Outputs.
    ----
        None. 
    """
    file = open(file_name,mode="w+")
    for val in labels:
        file.write(val+',')
    file.write('\n')
    h,w = array.shape
    for i in range(h):
        for j in range(w):
            file.write(str(array[i,j]+',')
        file.write('\n')
    file.close()
