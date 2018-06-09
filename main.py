from torchvision import datasets
from torchvision import transforms

import torch.optim
from torch import nn
import torch.utils.data

import network


def main ():
	nameOfTest = input ( "What is the name of the test? " )

	transformationsAddMore = transforms.Compose ( [
		transforms.RandomRotation ( degrees = 15 ),
		transforms.ToTensor ()
	] )

	trainingDataset = torch.utils.data.ConcatDataset ( [
		datasets.MNIST ( root = 'data/train', transform = transformationsAddMore, download = True ),
		datasets.MNIST ( root = 'data/train', transform = transformationsAddMore ),
		datasets.MNIST ( root = 'data/train', transform = transformationsAddMore )
	] )
	testingDataset = datasets.MNIST ( root = 'data/test', transform = transformationsAddMore,
	                                  train = False, download = True )

	trainingDataLoader = torch.utils.data.DataLoader ( trainingDataset,
	                                                   batch_size = 32,
	                                                   shuffle = True )
	testingDataLoader = torch.utils.data.DataLoader ( testingDataset,
	                                                  batch_size = len ( testingDataset ) )

	testingResults = []

	for i in range ( 5 ):

		net = network.Network ()

		optimizer = torch.optim.SGD ( net.parameters (), lr = 1e-02 )

		lossFunction = nn.CrossEntropyLoss ()

		print ( "Round {}".format ( i + 1 ) )
		print ( "Training!" )
		net.train ()
		for epoch in range ( 0, 13 ):
			print ( "Epoch {}".format ( epoch ) )

			for batchNum, (data, target) in enumerate ( trainingDataLoader ):

				optimizer.zero_grad ()

				output = net ( data )

				loss = lossFunction ( output, target )
				loss.backward ()

				optimizer.step ()

				if batchNum % 50 == 0:
					print (
							"Training loss at epoch {} batch number {}: {}".format ( epoch,
							                                                         batchNum,
							                                                         loss ) )

		print ( "Testing!" )
		net.eval ()
		correct = 0
		for _, (data, target) in enumerate ( testingDataLoader ):
			output = net ( data )

			pred = output.max ( 1, keepdim = True )[1]  # get the index of the max log-probability
			correct += pred.eq ( target.view_as ( pred ) ).sum ().item ()
		print ( "Number of correctness is {}".format ( correct ) )
		print (
				"Percentage of correctness is {}%".format (
						correct / len ( testingDataset ) * 100. ) )
		testingResults.append ( float ( correct ) / float ( len ( testingDataset ) ) )
	resultsFile = open ( "results.txt", mode = "a+" )
	resultsFile.write ( "\n" + nameOfTest + ": " + str (
			float ( sum ( testingResults ) ) / float ( len ( testingResults ) ) * 100 ) + "%" )
	resultsFile.close ()


if __name__ == '__main__':
	main ()
