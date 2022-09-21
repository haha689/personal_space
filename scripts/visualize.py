import torch 
import os
import matplotlib.pyplot as plt
path = os.path.join(os.getcwd(), 'checkpoints',
                                    '%s_with_model.pt' % "checkpoint0_fixed")
checkpoint = torch.load(path, map_location=torch.device('cpu'))
#plt.plot(checkpoint['sample_ts'], checkpoint['metrics_train'], label = "train")
plt.plot(checkpoint['sample_ts'], checkpoint['metrics_val'], label = "val")
plt.xlabel("num of iterations")
plt.ylabel("loss")
plt.title("model trained on [1,2,3,4] & evaled on [0]")
plt.ylim(0, 10**2)
plt.legend()
plt.show()