import argparse





# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(prog = 'LABEL', add_help=True)
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m","--model", required=True,
	help="path to the model in .h5 format")
ap.add_argument("--info", required=False, default=False,
	help="Show a summary of the model")
ap.add_argument("--plot_model", required=False, default=False,
	help="Save an image of the model")
ap.add_argument("--plot_name", required=False, default="model.png",
	help="Normalize the data from 0-255 to 0-1")
ap.add_argument("-n","--normalize", required=False, default=True,
	help="Name of the model")
ap.add_argument("-p","--probabilities", required=False, default=False,
	help="Show the probabilities")
args = vars(ap.parse_args())
