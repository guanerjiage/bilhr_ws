# Tutorial 3

## About mnist dataset

mnist_fnn.py contains the training and test for mnist dataset. We achieved about 94% accuracy.

## To test the robot's performance:

1. Open the initial nao robot environment in webot.
2. Set robot translation to x=-1m, y default, z default.
3. Manuelly rotate the robot around y axis, 90 degree clockwise, the rotation property should be then: 0.577, 0.577, 0.577, -2.094.
4. Import the Solid.wbo object, you should see a red ball in front of the robot. You can use the three axis on the ball to move it.
5. Run central_node.py, be sure the terminal path in now the folder with track_network_i.txt, otherwise there will be an error like "file not found".
6. Run keyboard_node.py, first type a and then b to move the head and arm to home position, then type e to set robot's state to track the red ball.
7. Then move the ball, the robot might take 2 or 3 moves to reach the target, since it might use the red blob position during the ball movement, or the ball might be covered by the arm at some time point. And the arm cannot reach the ball near the right edge of the image.

## Training data

- The raw training data are saved in training_x.txt and training_y.txt. training_x.txt contains the pixel of red centers. First column is x coordinate, second column is y coordinate. training_y.txt contains the value for left shoulder pitch and roll, first column pitch, second column roll. training_x_norm.txt and training_y_norm.txt contain normalized data. But only normalized pixel coordinates are used. The network directly outputs the pitch and roll without further processing.
- Trained network parameters are already saved in track_newtork_i.txt. Execute track_fnn.py to train new paramters. If you want to change the network structure, don't forget to change both central_node.py and track_fnn.py.
