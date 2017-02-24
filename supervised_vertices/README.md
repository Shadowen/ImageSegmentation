# Working CNN
2 February 2017

The prediction layer of the CNN looks very promising. Before the softmax, you can see that the prediction is most confident at the target (the next point), but it is also somewhat confident along the entire line towards the answer. Hopefully I can come up with a cost function or softmax layer that does not punish this behaviour!

One concern I have is that the training set is too large and may represent a significant fraction of the input space, such that overfitting on the training set still results on good performance on the held out validation set. I will have to increase the image size (currently 32x32) and complexity (pixels are 0 or 1 right now, monochrome, convex shapes only) to find out if this is true.

I also went ahead and added a little bit of noise (4% random pixels) to the images. I will introduce more complex polygons later.

# First end-to-end system integration
4 February 2017

If the network picks a point within 2 pixels (6% of the image) of the starting point, the shape is automatically closed. This only works about 90% of the time and gives an IOU in the 60-70% (70-80% without noise) range.

### Observations
The neural net actually is better at finding corners than ordering corners (unsurprisingly). See perfect_iou.png.

This means that sometimes it skips a corner and cuts off a chunk of the shape. See shallow_angle.png or skip_corner.png attached.

Sometimes the network forgets the direction it is traversing the shape and goes backwards. This is probably related, but usually causes bigger issues. See wrong_direction.png.

The higher scores (> 80 IOU) will trend higher with higher resolution images, because of aliasing occuring along edges.

In the long term, it is probably worth investigating if I can use a separate output neuron (share most of the network except the output layers) to indicate the end of the polygon.


IOU for valid shapes only is ~80% even on noised and blurred dataset.


# History mask
Experiments with zeroing out the history mask show the history mask is unused
   - will need to visualize the weights to confirm
   - try some sort of regularization to prevent this?
   
## Meeting notes
8 February 2017
- Try two points as history mask
- Show training set IOU and validation set IOU together
- Try reducing vertices with angles ~180 degrees
- Distance Transform


# History experimentation
On validation set of 1000 images
line-mask:
IOU = 0.8049418066159604, failed = 129

line mask, zeroed:
IOU = 0.791100085984716, failed = 237

Line history (fixed):
IOU = 0.8478980740994885, failed = 72

Line history (fixed, zeroed):
IOU = 0.8266719020871185, failed = 145


No history mask:
IOU = 0.8406220083067095, failed = 194


Point history:
IOU = 0.8219838139802708, failed = 139

Point history (zeroed):
IOU = 0.832699836324291, failed = 208

__Conclusion:__ the history mask does help reduce failures to close the shape. The line history mask is much better than the point history mask for this purpose.

- Errors where the network appears to go in the wrong direction are because a single pixel is higher than the next point
- Try spatial softmax? - convolution layers
    - Add a "valid moves" mask? multiply it in as a fixed constant near the last layer?

# Validity mask

- ~~Try using a mask that indicates which points are valid outputs.~~

Validity mask (line history, fixed):
IOU = 0.760096910805428, failed = 118

Results for the validity mask are inconclusive. Need to refine the validity mask implementation with some sort of ray tracing algorithm. TODO

## Meeting notes
- Smooth the ground truth with soft crossentropy
- ~~Use an RNN~~
- Try providing the point required to close the shape as an input

# Recurrent Neural Network
## Working RNN
- Working CNN + RNN implementation; cross entropy is decreasing over at least 10k steps, but accuracy doesn't seem to be getting better

### TODO
- Add the image summaries back in
- Add validation set measurements to rnn.py to see if it's working?
- Run overnight
- Try providing the first (final) point as input
- "Upgrade" to multi-layer LSTM
- Need to achieve 90%+ on simple polygons
- Get running on the cluster
- Test on Luis' dataset
- Reinforcement learning (A3C)
