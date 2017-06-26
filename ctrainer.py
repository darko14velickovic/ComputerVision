
from image_processor.Trainer import CnnTrainer
import glob, os
import sys, getopt
import cv2 as cv
import numpy as np

def main(argv):
   networkName = 'default'
   usage_string = 'ctrainer.py -d <image dimension DIM x DIM> -n <network name> -f <comma separated names of folders' \
                  'for training and testing> -ec <number of training images> -tc number of testing images'

   dimension = 40
   folders = []
   example_count = []
   test_count = 5

   try:
      opts, args = getopt.getopt(argv,"hd:n:f:e:t:")
   except getopt.GetoptError:
       print (usage_string)
       sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print (usage_string)
         sys.exit(0)
      elif opt in ("-d", "--dimension"):
          print arg
          dimension = int(arg)
      elif opt in ("-n", "--name"):
          networkName = arg
      elif opt in ("-f", "--folders"):
          folders = arg.split(',')
      elif opt in ("-e", "--examples"):
          string_lens = arg.split(',')
          for string_len in string_lens:
              example_count.append(int(string_len))
      elif opt in ("-t", "--tests"):
          test_count = int(arg)

   print ('Network name is ' + networkName)
   print ('Dimension of images is: ', dimension, ' x ', dimension)
   print ('Folders for training and testing are: ')
   print (folders)
   print ('Example count is: ', example_count)
   print ('Test count is: ', test_count)
   if example_count.__len__() != folders.__len__():
       print('Folders and test examples are different!')
       return -1

   trainer = CnnTrainer(dimension, networkName, False)

   trainer.tf_learn(networkName, folders, example_count, test_count)

# if __name__ == "__main__":
#    main(sys.argv[1:])


def mergeInOneImage(dir, pattern):
    imageNumber = 0
    imageRow = 0
    bigImage = np.zeros()
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        # title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        image = cv.imread(pathAndFilename)
        if imageNumber == 19:
            imageRow += 1
            imageNumber = 0



        imageNumber += 1

#
def rename(dir, pattern):
    counter = 1
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename,
                  os.path.join(dir, "sjiodkkdka" + str(counter) + ext))
        counter += 1
rename(r'/home/darko/Documents/ComputerVision/training data/other/', r'*.png')
rename(r'/home/darko/Documents/ComputerVision/training data/point/', r'*.png')
rename(r'/home/darko/Documents/ComputerVision/training data/palm', r'*.png')
rename(r'/home/darko/Documents/ComputerVision/training data/fist', r'*.png')