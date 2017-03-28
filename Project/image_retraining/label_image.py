import tensorflow as tf, sys
import time

# Prints out the time spent on a given function
def print_timing (func):
  def wrapper (*arg):
    t1 = time.time ()
    res = func (*arg)
    t2 = time.time ()
    print ("--->{} took {} ms".format (func.__name__, (t2 - t1) * 1000.0))
    return res
  return wrapper

image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels_42signs.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph_42signs.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

@print_timing
def print_top(top_k,predictions):
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))

@print_timing
def predict(softmax_tensor,image_data):
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    print_top(top_k)



with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predict(softmax_tensor,image_data)
