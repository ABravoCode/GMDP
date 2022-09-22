import torch
import numpy as np

from forest.victims.victim_single import _VictimSingle
import forest

sigma = 0.1
args = forest.options().parse_args()

if __name__ == '__main__':
	orig_net = _VictimSingle(args)
	orig_net.load_state_dict(torch.load(r"CIFAR10_['ResNet18']_conservative_clean_model.pth", map_location=torch.device('cpu')))
	d = 0
	j = 0
	all_param = []
	len_param = []
	all_shape = []
	all_name = []
	for name, param in orig_net.named_parameters():
		d += param.numel()
		j += param.numel()
		all_param.append(param)
		len_param.append(j)
		all_shape.append(param.shape)
		all_name.append(name)
		# print(param)

	len_param = list(map(lambda x:x-1, len_param))
	len_param.insert(0, 0)

	delta = np.random.normal(0, sigma, (d))
	
	temp = all_param[0]
	for index, param in enumerate(all_param):
		temp = param.clone().detach().numpy()
		if index == 0:
			dx = delta[len_param[index]:len_param[index+1]+1] 
		else: 
			dx = delta[len_param[index]+1:len_param[index+1]+1] 
		temp = temp + dx.reshape(all_shape[index])
		param.data = torch.tensor(temp)

	net = _VictimSingle(args)
	net.load_state_dict(torch.load(r"CIFAR10_['ResNet18']_conservative_clean_model.pth", map_location=torch.device('cpu')))

	for index, name in enumerate(all_name):
		net[name] = all_param[index-1]

	torch.save(orig_net.state_dict(), r"delta_CIFAR10_['ResNet18']_conservative_clean_model.pth")
	
	# for name, param in orig_net.named_parameters():
	# 	print(param)

	new_net = _VictimSingle(args)
	new_net.load_state_dict(torch.load(r"delta_CIFAR10_['ResNet18']_conservative_clean_model.pth", map_location=torch.device('cpu')))
