import caffe

# Load the model
net = caffe.Net('C:/Users/Dheeraj/OneDrive/Desktop/New folder (4)/model/gender_deploy.prototxt', 'C:/Users/Dheeraj/OneDrive/Desktop/New folder (4)/model/gender_net.caffemodel', caffe.TEST)

# Perform inference or other operations
# ...
