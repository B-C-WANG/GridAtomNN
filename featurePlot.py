import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

data = np.load("trainX.npy")



print(data.shape)
fig = plt.figure()

def update(i):
    return plt.imshow(data[0,i,:,:,0]) # different height
def update1(i):
    print(i)
    return plt.imshow(data[i,0,:,:,0]) # different sample

ani = animation.FuncAnimation(fig, update,frames=np.arange(0,data.shape[1]))
#ani.save('test.mp4', fps=15,extra_args=['-vcodec', 'libx264'],writer='ffmpeg_file')
ani.save('line.gif', dpi=80, writer='imagemagick')
#plt.show()





