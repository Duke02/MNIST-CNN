from torchvision import datasets
from torchvision import transforms

import torch.optim
from torch import nn
import torch.utils.data

import network


def main ():
	net = network.Network ()

	transformations = transforms.Compose ( [
		transforms.ToTensor ()
	] )
	trainingDataset = datasets.MNIST ( root = 'data/train', transform = transformations,
	                                   download = True )
	testingDataset = datasets.MNIST ( root = 'data/test', transform = transformations,
	                                  train = False, download = True )

	trainingDataLoader = torch.utils.data.DataLoader ( trainingDataset,
	                                                   batch_size = 16,
	                                                   shuffle = True )
	testingDataLoader = torch.utils.data.DataLoader ( testingDataset,
	                                                  batch_size = len ( testingDataset ) )

	optimizer = torch.optim.SGD ( net.parameters (), lr = 1e-02 )
	scheduler = torch.optim.lr_scheduler.StepLR ( optimizer = optimizer, gamma = 0.1,
	                                              step_size = 4 )

	lossFunction = nn.CrossEntropyLoss ()

	print ( "Training!" )
	net.train ()
	for epoch in range ( 0, 13 ):
		print ( "Learning rate is {}".format ( scheduler.get_lr () ) )
		print ( "Epoch {}".format ( epoch ) )

		for batchNum, (data, target) in enumerate ( trainingDataLoader ):

			optimizer.zero_grad ()

			output = net ( data )

			loss = lossFunction ( output, target )
			loss.backward ()

			optimizer.step ()

			if batchNum % 50 == 0:
				print ( "Training loss at epoch {} batch number {}: {}".format ( epoch, batchNum,
				                                                                 loss ) )
		scheduler.step ( epoch = epoch )

	print ( "Testing!" )
	net.eval ()
	correct = 0
	for _, (data, target) in enumerate ( testingDataLoader ):
		output = net ( data )

		pred = output.max ( 1, keepdim = True )[1]  # get the index of the max log-probability
		correct += pred.eq ( target.view_as ( pred ) ).sum ().item ()
	print ( "Number of correctness is {}".format ( correct ) )
	print ( "Percentage of correctness is {}%".format ( correct / len ( testingDataset ) * 100. ) )


if __name__ == '__main__':
	main ()
