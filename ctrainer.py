
from image_processor.Trainer import CnnTrainer

import sys, getopt

def main(argv):
   networkName = 'default'
   dimension = 40

   try:
      opts, args = getopt.getopt(argv,"hd:n:")
   except getopt.GetoptError:
       print ('ctrainer.py [-d | --dimension] <image dimension DIM x DIM> [-n | --name] <network name>')
       sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print ('ctrainer.py -d <image dimension DIM x DIM> -n <network name>')
         sys.exit(0)
      elif opt in ("-d", "--dimension"):
          print arg
          dimension = int(arg)
      elif opt in ("-n", "--name"):
          networkName = arg
   print ('Network name is ' + networkName)
   print ('Dimension of images is: ', dimension, ' x ', dimension)

   trainer = CnnTrainer(dimension, networkName, False)

   trainer.tf_learn(networkName)

if __name__ == "__main__":
   main(sys.argv[1:])