# HESS CSV File Move
# Palomino, Oct 23, 2017

def HESSfilemove(newFileName):

    # Import Modules
    import os

    prevFileName = newFileName
    newFileLoc = r'C:\Users\Alex\Box Sync\Alex and Masood\WestSmartEV\Study Locations\RMP North Temple Office\Prelim Data\Auto/'+str(newFileName)

    print(newFileLoc)

    # Move file to newFileLoc
    os.rename(prevFileName, newFileLoc)


