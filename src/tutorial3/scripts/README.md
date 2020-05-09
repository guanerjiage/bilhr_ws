mnist_fnn.py contains the training and test for mnist dataset. It achieved about 94% accuracy.

Set robot translation to x=-1m, y default, z default.
Manuelly rotate the robot around y axis, 90 degree clockwise, the rotation property should then be: 0.577, 0.577, 0.577, -2.094
Import the Solid.wbo object, you should see a red ball in front of the robot. You can use the three axis on the ball to move it

Run central_node and keyboard_node, first type a and then b to move the head and arm to home position, then type e to toggle robot's state to track the red ball.
Then move the ball, the robot might take 2 or 3 moves to reach the target, since it might use the red blob position during the ball movement, or the ball might be covered by the arm at some time point. And the arm cannot reach the ball at the right side of the image.

The training data are saved in training_x.txt and training_y.txt. training_x.txt contains the pixel of red centers. First column is x coordinate, second column is y coordinate. training_y.txt contains the should value for left shoulder pitch and roll, first column pitch, second column roll.

Trained network parameters are already saved in track_newtork_l.txt. Execute track_fnn.py to train new paramters. If you want to change the network structure, don't forget to change both central_node.py and track_fnn.py.
