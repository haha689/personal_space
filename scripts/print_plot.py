import matplotlib.pyplot as plt
import torch 
import os
num = 4
epoch = 200000
plot_path = os.path.join(os.getcwd(),"plots", "checkpoint%d_fixed2_%d_round_2.pt" % (num, epoch))
plot_dict = torch.load(plot_path, map_location=torch.device('cpu'))
#plt.plot(plot_dict['x'], plot_dict['y1'], "bo")
#plt.xlabel("norm of ground truth")
#plt.ylabel("prediction error")
#plt.title("checkpoint%d" %num)
#plt.ylim(0, 10**2)
#plt.show()
plt.plot(plot_dict['x'], plot_dict['y2'], "bo")
plt.xlabel("norm of ground truth")
plt.ylabel("prediction uncertainties")
plt.title("checkpoint%d" %num)
#plt.ylim(0, 10**2)
plt.show()