import torch 
import os
import matplotlib.pyplot as plt
path = os.path.join(os.getcwd(), 'checkpoints',
                                    '%s_with_model_%d.pt' % ("checkpoint3_fixed2", 270000))
checkpoint = torch.load(path, map_location=torch.device('cpu'))
print("20th mse_metric:")
print(checkpoint['mse_metric_train'][20])
plt.plot(checkpoint['sample_ts'], checkpoint['mse_metric_train'], label = "train")
plt.xlabel("num of iterations")
plt.ylabel("mse_metric")
plt.title("model trained on [0,1,3,4] & evaled on [2]")
#plt.ylim(0, 10**2)
plt.legend()
plt.show()