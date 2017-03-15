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
~~- Add the image summaries back in~~
~~- Add validation set measurements to rnn.py to see if it's working?~~
~~- "Upgrade" to multi-layer LSTM~~
~~- Run overnight~~
~~- Run on server~~

# 1 March 2017
- CNN + RNN
- Overnight run on server shows improvement in validation loss and error after ~5000 iterations
- Minimum of ~1.5 pixels mean validation error
- Pictures still show that the network is unsure which point is next to pick, but it no longer goes backwards and re-picks already chosen points

### TODO
~~- Debug accuracy measurement~~
Probably just a Python 2 problem; use `from __future__ import division` as temporary fix?
~~- Program IOU for RNN~~
95%+ training iou and approximately 75% validation IOU
- Check failure rate
- Investigate durations for d > 3
- Clean up IOU code - use Tensorflow variable sharing to build both the "training network" and "inference network"
- IOU image summaries

# 2 March 2017
- Working policy gradients implementation (on CartPole-v1)
- ~~Investigate why NaNs appear in the optimizer when changing TD-target to TD-error..?~~
Fixed - should just use softmax_cross_entropy_with_logits function which gets rid of singularities in the softmax

# 5 March 2017
- ~~Move the learning into the episode loop~~


# 10 March 2017
- A3C on RandomShapesVertices-v0 goes from ~23% to ~28% IOU over 10 hours
- Too slow, consider pretraining
- Very high entropy still, what's the random baseline?
- CNN can achieve down to 2% failure rate with ~80-85% IOU with new architecture/optimizer
~~- Advanced optimizers seem to be less stable?~~
    - Use smaller learning rate

# 13 March 2017
- ~~Talk to Tingwu~~
- Install Universe/Docker
- Get running on the cluster
    - Need to bring your own Python install? Ask Relu
- Test on Luis' dataset

### TODO
- Reinforcement learning (A3C)
    - ~~Debug the A3C algorithm. It seems to peak in performance much lower than the paper reported~~ Use the [Universe starter agent](https://github.com/openai/universe-starter-agent)
    - Pretrain the policy network
    - Running out of memory... :( try on the server
    - Add teacher to RL by inserting permanent samples into replay memory - remember this only works for off-policy methods!
- Try providing the first (final) point as input
- Need to achieve 90%+ on simple polygons
    

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%10 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})